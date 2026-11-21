"""The CLM Hypothesis Agent on a Hugging Face model (generation of rewrites + four-way scoring).

One HA turn for a batch of contexts:

1. (edit mode) the HA reads its context inside ``prompts/clm_hypothesis_agent.txt`` and replies
   ``KEEP`` or ``<rewrite>...</rewrite>``; a rewrite replaces the context (src/clm/context.py);
2. the hypothesis is read from the resulting context by scoring the four diseases after
   ``Hypothesis:`` in the usual HA prompt (``prompts/hypothesis_agent.txt``), exactly as
   ``src/score_ha.py`` and the frozen HA do.

Append-only mode skips step 1: then the agent is the H2 scorer on a growing record.

The model may be a plain model, a PEFT model with one adapter (``adapter_name=None``) or a
PEFT model where the HA is a second adapter next to the DA's (``adapter_name="clm_ha"``, stage
4). In the last case every call saves and restores the active adapter, the LoRA
``requires_grad`` flags and the training mode, as ``src/ha/frozen.py`` does.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from src.clm.context import APPEND_ONLY, TOO_LONG, CLMState, TurnRecord
from src.ha.prompts import ANSWER_PREFIX, DISEASES, candidate_ids, chat_prompt, encode
from src.ha.scoring import candidate_logprobs, softmax

LLAMA3_CONTEXT = 8192


class CLMHA:
    def __init__(self, model, tokenizer, score_template: str, edit_template: str, *, adapter_name: str | None = None,
                 edit: bool = True, max_new_tokens: int = 512, max_words: int = 250, temperature: float = 1.0,
                 max_total_tokens: int = LLAMA3_CONTEXT, diseases=DISEASES) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.score_template = score_template
        self.edit_template = edit_template
        self.adapter_name = adapter_name
        self.edit = edit
        self.max_new_tokens = max_new_tokens
        self.max_words = max_words
        self.temperature = temperature
        self.max_total_tokens = max_total_tokens
        self.diseases = tuple(diseases)
        self.candidates = candidate_ids(tokenizer, self.diseases)
        self.answer_prefix = encode(tokenizer, ANSWER_PREFIX)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        stops = {tokenizer.eos_token_id}
        for token in ("<|eot_id|>", "<|end_of_text|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                stops.add(token_id)
        self.stop_ids = sorted(i for i in stops if i is not None)

    # ------------------------------------------------------------------ prompts
    def score_prompt_ids(self, context: str) -> list[int]:
        message = self.score_template.format(provided_test_results=context)
        return encode(self.tokenizer, chat_prompt(self.tokenizer, message)) + self.answer_prefix

    def edit_prompt_ids(self, context: str) -> list[int]:
        message = self.edit_template.format(context=context, max_words=self.max_words)
        return encode(self.tokenizer, chat_prompt(self.tokenizer, message))

    def label_ids(self, label: str) -> list[int]:
        return self.candidates[self.diseases.index(label.lower())]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        if self.adapter_name is None:
            return [p for p in self.model.parameters() if p.requires_grad]
        return [p for n, p in self.model.named_parameters() if f".{self.adapter_name}." in n]

    # ------------------------------------------------------------- model state
    def _switch(self):
        """Activate the HA adapter; return a function that restores the previous state."""
        model = self.model
        was_training = model.training
        if self.adapter_name is None:
            return lambda: model.train(was_training)
        active = model.active_adapter
        flags = [(p, p.requires_grad) for n, p in model.named_parameters() if "lora_" in n]
        model.set_adapter(self.adapter_name)

        def restore():
            model.set_adapter(active)
            for p, flag in flags:
                p.requires_grad_(flag)
            model.train(was_training)

        return restore

    @contextmanager
    def inference(self):
        restore = self._switch()
        self.model.eval()
        try:
            yield
        finally:
            restore()

    @contextmanager
    def training(self):
        """HA adapter active and trainable, model in train mode (dropout, checkpointing)."""
        restore = self._switch()
        if self.adapter_name is not None:
            for p in self.trainable_parameters():
                p.requires_grad_(True)
        self.model.train()
        try:
            yield
        finally:
            restore()

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    # --------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompts: list[list[int]], sample: bool) -> list[list[int]]:
        """Completions (token ids, up to and including the first stop token) for each prompt."""
        if not prompts:
            return []
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[self.pad_id] * (width - len(p)) + p for p in prompts], device=self.device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=self.device)
        kwargs = dict(max_new_tokens=self.max_new_tokens, eos_token_id=self.stop_ids, pad_token_id=self.pad_id,
                      use_cache=True)
        if sample:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self.inference():
            out = self.model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        completions = []
        for row in out[:, width:].tolist():
            cut = next((i for i, t in enumerate(row) if t in self.stop_ids), None)
            completions.append(row if cut is None else row[: cut + 1])
        return completions

    @torch.no_grad()
    def score(self, contexts: list[str]) -> list[tuple[list[int], list[float]]]:
        """(scoring prompt ids, probabilities in ``self.diseases`` order) for each context."""
        out = []
        with self.inference():
            for context in contexts:
                ids = self.score_prompt_ids(context)
                out.append((ids, softmax(candidate_logprobs(self.model, ids, self.candidates))))
        return out

    # --------------------------------------------------------------------- turn
    def turn(self, states: list[CLMState], sample: bool = False, batch_size: int = 8) -> list[TurnRecord]:
        """One HA turn for each state (in place): optional rewrite, then scoring."""
        actions = [APPEND_ONLY] * len(states)
        edits: dict[int, tuple[list[int], list[int], str]] = {}
        if self.edit:
            todo = []
            for i, state in enumerate(states):
                ids = self.edit_prompt_ids(state.context)
                if len(ids) + self.max_new_tokens > self.max_total_tokens:
                    actions[i] = TOO_LONG
                else:
                    todo.append((i, ids))
            for start in range(0, len(todo), batch_size):
                chunk = todo[start: start + batch_size]
                for (i, ids), completion in zip(chunk, self.generate([ids for _, ids in chunk], sample)):
                    text = self.tokenizer.decode(completion, skip_special_tokens=True)
                    actions[i] = states[i].apply(text)
                    edits[i] = (ids, completion, text)
        records = []
        for i, (ids, probs) in enumerate(self.score([s.context for s in states])):
            prompt, completion, text = edits.get(i, ([], [], ""))
            records.append(TurnRecord(action=actions[i], context=states[i].context, probs=probs,
                                      context_tokens=len(ids), score_prompt_ids=ids, edit_prompt_ids=prompt,
                                      completion_ids=completion, completion=text))
        return records
