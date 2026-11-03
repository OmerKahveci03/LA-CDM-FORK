"""Evidence Critic: prompt construction, schema-validated parsing, local backend.

The critic sees only the current decision point: the history summary, results of
tests already requested, the current hypothesis/confidence, audit flags and the
tests that may still be requested. It never sees the label, unrequested test
results, or later steps.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from src.trace_cdm.schemas import CRITIC_NEXT_STEPS, AuditResult, CriticResult, StepState

PROMPT_HEADER = """You are an evidence auditor supporting a diagnostic agent for patients with abdominal pain. You are NOT the diagnosing clinician and you must not diagnose.
Review ONLY the information below. Do not invent test results, findings or patient facts. Identify weaknesses in the evidence behind the current hypothesis: equivocal or missing findings, contradictions between results, and claims the evidence does not support.

Respond with a single JSON object and nothing else, using exactly these keys:
{"evidence_concerns": [strings], "unsupported_claims": [strings], "contradictions": [strings], "recommended_next_step": "proceed" | "request_test" | "review", "recommended_test": one test name from REQUESTABLE TESTS or null, "reason": short string, "evidence_quality": number from 0 to 1}
Use "request_test" only with a test from REQUESTABLE TESTS. Keep every string under 25 words."""


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " [truncated]"


def build_prompt(state: StepState, audit: AuditResult, max_result_chars: int = 1200) -> str:
    lines = [PROMPT_HEADER, "", "PATIENT HISTORY SUMMARY:", _truncate(state.patient_history, max_result_chars), ""]
    lines.append("TEST RESULTS SO FAR:")
    if state.evidence:
        for item in state.evidence:
            body = _truncate(item.text, max_result_chars) if item.available else "not available"
            lines.append(f"- {item.name} ({item.source}): {body}")
    else:
        lines.append("- none yet")
    lines.append("")
    if state.hypothesis is not None:
        conf = f"{state.confidence * 10:.0f}/10" if state.confidence is not None else "not given"
        lines.append(f"CURRENT HYPOTHESIS: {state.hypothesis} (confidence {conf}, self-reported, not calibrated)")
    rule_flags = [f.message for f in audit.flags if f.severity != "info"]
    lines.append("RULE-BASED FLAGS: " + ("; ".join(rule_flags) if rule_flags else "none"))
    lines.append("REQUESTABLE TESTS: " + (", ".join(state.remaining_tests) if state.remaining_tests else "none"))
    lines.append("")
    lines.append("JSON:")
    return "\n".join(lines)


def _string_list(value, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{key} must be a list of strings")
    return [v.strip() for v in value if v.strip()]


def parse_critic(text: str, remaining_tests: list[str]) -> CriticResult:
    """Validate critic output. Any malformed output falls back to an empty review."""
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("JSON is not an object")
        result = CriticResult(
            evidence_concerns=_string_list(data.get("evidence_concerns"), "evidence_concerns"),
            unsupported_claims=_string_list(data.get("unsupported_claims"), "unsupported_claims"),
            contradictions=_string_list(data.get("contradictions"), "contradictions"),
            recommended_next_step=str(data.get("recommended_next_step", "")).strip().lower(),
            recommended_test=data.get("recommended_test"),
            reason=str(data.get("reason") or "").strip(),
        )
        if result.recommended_next_step not in CRITIC_NEXT_STEPS:
            raise ValueError(f"invalid recommended_next_step {result.recommended_next_step!r}")
        quality = data.get("evidence_quality")
        if isinstance(quality, (int, float)) and not isinstance(quality, bool) and 0 <= quality <= 1:
            result.evidence_quality = float(quality)
    except (ValueError, TypeError) as exc:
        return CriticResult(parse_error=str(exc))
    return validate_recommendation(result, remaining_tests)


def validate_recommendation(result: CriticResult, remaining_tests: list[str]) -> CriticResult:
    """Keep a recommended test only if it can still be requested in this episode."""
    by_lower = {t.lower(): t for t in remaining_tests}
    test = result.recommended_test
    if isinstance(test, str) and test.strip().lower() in by_lower:
        result.recommended_test = by_lower[test.strip().lower()]
    elif test not in (None, "", "null"):
        result.recommended_test = None
        result.invalid_recommendation = True
    else:
        result.recommended_test = None
    if result.recommended_next_step == "request_test" and result.recommended_test is None:
        result.recommended_next_step = "review"
        result.invalid_recommendation = True
    return result


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class LocalCritic:
    """Batched greedy generation from a local Hugging Face checkpoint.

    Loads with ``local_files_only=True``; nothing is downloaded. Identical prompts
    are served from an in-memory cache.
    """

    def __init__(self, model_path: str, device: str = "cuda:2", load_in_4bit: bool = False,
                 max_new_tokens: int = 256, batch_size: int = 8):
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Critic checkpoint directory not found: {path}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if device.startswith("cuda:") and int(device.split(":")[1]) >= torch.cuda.device_count():
            raise RuntimeError(
                f"Critic device {device} requested but only {torch.cuda.device_count()} GPU(s) are visible. "
                "Request another GPU or set trace_cdm.critic.device."
            )
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": device},
        ).eval()
        self.model_id = str(path)
        self.cache: dict[str, str] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "generation_seconds": 0.0}

    def _format(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, prompts: list[str]) -> list[tuple[str, dict]]:
        """Return ``(text, call_info)`` per prompt, where call_info has latency/tokens/cached."""
        results: list[tuple[str, dict] | None] = [None] * len(prompts)
        todo = []
        for i, prompt in enumerate(prompts):
            key = prompt_hash(prompt)
            if key in self.cache:
                self.stats["cache_hits"] += 1
                results[i] = (self.cache[key], {"cached": True, "latency_seconds": 0.0,
                                                "prompt_tokens": 0, "completion_tokens": 0})
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            texts = [self._format(prompts[i]) for i in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.pad_token_id)
            elapsed = time.perf_counter() - t0
            new_tokens = out[:, enc.input_ids.shape[1]:]
            for row, i in enumerate(chunk):
                text = self.tokenizer.decode(new_tokens[row], skip_special_tokens=True)
                n_prompt = int(enc.attention_mask[row].sum())
                n_completion = int((new_tokens[row] != self.tokenizer.pad_token_id).sum())
                self.cache[prompt_hash(prompts[i])] = text
                self.stats["prompt_tokens"] += n_prompt
                self.stats["completion_tokens"] += n_completion
                results[i] = (text, {"cached": False, "latency_seconds": elapsed / len(chunk),
                                     "prompt_tokens": n_prompt, "completion_tokens": n_completion})
            self.stats["generation_seconds"] += elapsed
        self.stats["calls"] += len(prompts)
        return results

    def resource_stats(self) -> dict:
        stats = dict(self.stats)
        if self.device.startswith("cuda"):
            stats["peak_gpu_memory_gb"] = self.torch.cuda.max_memory_allocated(self.device) / 1e9
        stats["model"] = self.model_id
        return stats


def load_critic(cfg: dict) -> LocalCritic:
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("trace_cdm.critic.model_path (or TRACE_CRITIC_MODEL) must name a local checkpoint")
    logging.info("Loading TRACE critic from %s on %s", model_path, cfg.get("device"))
    return LocalCritic(
        model_path,
        device=cfg.get("device", "cuda:2"),
        load_in_4bit=cfg.get("load_in_4bit", False),
        max_new_tokens=cfg.get("max_new_tokens", 256),
        batch_size=cfg.get("batch_size", 8),
    )
