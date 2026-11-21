"""H4: the CLM Hypothesis Agent (src/clm/). Pure parts always; model parts with a tiny Llama."""

import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

if "trl" not in sys.modules:
    trl = types.ModuleType("trl")
    data_utils = types.ModuleType("trl.data_utils")
    data_utils.maybe_apply_chat_template = lambda example, tokenizer: example
    trl.data_utils = data_utils
    sys.modules["trl"] = trl
    sys.modules["trl.data_utils"] = data_utils

from src.clm.context import (  # noqa: E402
    KEEP, MALFORMED, REWROTE, CLMState, TurnRecord, append_only_context, blocks_for, combined_advantages,
    fit_tests, outcome_advantages, parse_completion, random_sequence, success_gated_efficiency,
)
from src.ha.prompts import DISEASES, ha_user_message  # noqa: E402

TEMPLATE = "Info:\n{provided_test_results}"
EDIT = "Edit ({max_words} words):\n{context}\nEND"
PATIENT = {
    "patient_id": 7, "condition": "Pancreatitis", "patient_history": "epigastric pain",
    "test_results": {"Complete Blood Count": "WBC 15\n", "CT": "inflamed pancreas\n", "MRI": "not available.\n"},
}


class ContextTest(unittest.TestCase):
    def test_append_only_equals_h2_prompt_text(self):
        for tests in ([], ["ct"], ["CT", "complete blood count"]):
            expected = ha_user_message(TEMPLATE, PATIENT["patient_history"], PATIENT["test_results"], tests)
            self.assertEqual(TEMPLATE.format(provided_test_results=append_only_context(PATIENT, tests)), expected)

    def test_rewrite_then_append(self):
        state = CLMState.start("pain")
        state.observe("CT: x")
        self.assertEqual(state.apply("ok <rewrite>\n notes \n</rewrite>"), REWROTE)
        self.assertEqual(state.context, "notes")
        state.observe("MRI: y")
        self.assertEqual(state.context, "notes\n\nMRI: y")
        self.assertEqual(state.rewrites, 1)
        self.assertEqual(state.apply("KEEP"), KEEP)
        self.assertEqual(state.context, "notes\n\nMRI: y")

    def test_parse_completion(self):
        self.assertEqual(parse_completion("KEEP"), (KEEP, None))
        self.assertEqual(parse_completion("<rewrite>a</rewrite>"), (REWROTE, "a"))
        self.assertEqual(parse_completion("<REWRITE>a\nb</Rewrite>"), (REWROTE, "a\nb"))
        self.assertEqual(parse_completion("<rewrite>never closed"), (MALFORMED, None))
        self.assertEqual(parse_completion("<rewrite>  </rewrite>"), (MALFORMED, None))

    def test_blocks_use_original_names(self):
        self.assertEqual(blocks_for(PATIENT, ["ct", "complete blood count"]),
                         ["CT: inflamed pancreas\n", "Complete Blood Count: WBC 15\n"])

    def test_sequences_and_budget(self):
        rng = random.Random(0)
        for _ in range(20):
            seq = random_sequence(PATIENT, rng)
            self.assertTrue(1 <= len(seq) <= 2 and "MRI" not in seq)
        self.assertEqual(random_sequence(PATIENT | {"test_results": {"MRI": "not available.\n"}}, rng), [])
        kept = fit_tests(len, TEMPLATE, PATIENT, ["CT", "Complete Blood Count"], 60, random.Random(0))
        self.assertLessEqual(len(ha_user_message(TEMPLATE, "epigastric pain", PATIENT["test_results"], kept)), 60)
        self.assertLess(len(kept), 2)


class AdvantageTest(unittest.TestCase):
    def test_outcome(self):
        self.assertEqual(outcome_advantages([1, 1, 1, 1]), [0.0] * 4)
        adv = outcome_advantages([1, 0, 0, 0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertGreater(adv[0], 0)

    def test_success_gated(self):
        self.assertEqual(success_gated_efficiency([10, 20, 30, 40], [True, False, False, False]), [0.0] * 4)
        eff = success_gated_efficiency([100, 300, 50, 10], [True, True, False, False])
        self.assertEqual(eff[2:], [0.0, 0.0])  # failures get nothing, however cheap
        self.assertAlmostEqual(eff[0], 0.5)    # mean of successes 200: (200 - 100) / 200
        self.assertAlmostEqual(eff[1], -0.5)
        self.assertEqual(success_gated_efficiency([1, 1000], [True, True])[1], -0.998001998001998)
        combined = combined_advantages([1, 1, 0, 0], [100, 300, 50, 10], [True, True, False, False], 0.5)
        self.assertAlmostEqual(combined[0] - combined[1], 0.5)


class StubAgent:
    """Gives fixed probabilities and records the contexts it saw (environment tests)."""

    diseases = DISEASES

    def __init__(self, probs, rewrite_to=None):
        self.probs, self.rewrite_to, self.contexts = probs, rewrite_to, []
        self.model = SimpleNamespace(training=True)

    def turn(self, states, sample=False, batch_size=8):
        records = []
        for state in states:
            action = "keep"
            if self.rewrite_to is not None:
                state.apply(f"<rewrite>{self.rewrite_to}</rewrite>")
                action = "rewrite"
            self.contexts.append(state.context)
            records.append(TurnRecord(action=action, context=state.context, probs=list(self.probs),
                                      context_tokens=len(state.context), score_prompt_ids=[len(state.context)],
                                      completion_ids=[1, 2]))
        return records


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        from src.environment.environment import Environment, Trajectory

        self.Trajectory = Trajectory
        self.base = lambda: Environment("{patient_history}{first_hypothesis}", 4096, 13, list(DISEASES),
                                        ["Complete Blood Count", "CT", "MRI"], generate_hypothesis=True,
                                        hypothesis_prompt_template=TEMPLATE, generate_confidence_calibration=True)

    def trajectory(self, tests=("ct",)):
        t = self.Trajectory("x{first_hypothesis}", PATIENT["test_results"], "epigastric pain")
        t.provided_tests = list(tests)
        return t

    def test_same_transcript_as_frozen_ha(self):
        from src.clm.environment import CLMHAEnvironment
        from src.environment.frozen_ha_environment import FrozenHAEnvironment

        probs = [0.1, 0.1, 0.1, 0.7]
        clm = CLMHAEnvironment(self.base(), StubAgent(probs))
        frozen = FrozenHAEnvironment(self.base(), lambda m: probs, lambda m: [1], DISEASES)
        a, b = self.trajectory(), self.trajectory()
        clm._generate_hypothesis([a], None, None, None)
        frozen._generate_hypothesis([b], None, None, None)
        self.assertEqual(a.text, b.text)
        self.assertIn("The current hypothesis is: pancreatitis (confidence: 7)", a.text)
        self.assertEqual(a.ha_probs, [probs])

    def test_new_tests_are_appended_once_and_errors_keep_hypothesis(self):
        from src.clm.environment import CLMHAEnvironment

        agent = StubAgent([0.7, 0.1, 0.1, 0.1])
        env = CLMHAEnvironment(self.base(), agent)
        t = self.trajectory(tests=())
        env._generate_hypothesis([t], None, None, None)
        self.assertEqual(agent.contexts[-1], "Patient History: epigastric pain\n\n")
        t.provided_tests.append("ct")
        env._generate_hypothesis([t], None, None, None)
        self.assertEqual(agent.contexts[-1], "Patient History: epigastric pain\n\nCT: inflamed pancreas\n")
        t.error = True  # e.g. an unavailable test: no HA call
        env._generate_hypothesis([t], None, None, None)
        self.assertEqual(len(agent.contexts), 2)
        self.assertEqual(len(t.clm_turns), 2)
        t.provided_tests.append("complete blood count")
        env._generate_hypothesis([t], None, None, None)
        self.assertTrue(agent.contexts[-1].endswith("CT: inflamed pancreas\n\n\nComplete Blood Count: WBC 15\n"))

    def test_buffer_groups_and_notes_log(self):
        from src.clm.environment import CLMHAEnvironment

        with tempfile.TemporaryDirectory() as tmp:
            env = CLMHAEnvironment(self.base(), StubAgent([0.1, 0.1, 0.1, 0.7], rewrite_to="short notes"),
                                   notes_log_path=str(Path(tmp) / "notes.jsonl"))
            for run in (1, 2):
                env._run_id, env._label, env._patient_id, env._sample = run, "pancreatitis", run, True
                ts = [self.trajectory(), self.trajectory()]
                env._generate_hypothesis(ts, None, None, None)
                for t in ts:
                    t.request_count, t.diagnosis, t.invalid = 1, "pancreatitis", False
                kwargs = env._collect_reward_function_kwargs(ts)
                self.assertEqual(kwargs[0]["ha_diseases"], list(DISEASES))
            groups = env.take_groups()
            self.assertEqual([len(g) for g in groups], [2, 2])
            self.assertEqual(env.buffer, [])
            lines = [json.loads(l) for l in (Path(tmp) / "notes.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0]["turns"][0]["context"], "short notes")
            self.assertEqual(lines[0]["mode"], "train")


try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None


class CharTokenizer:
    """ASCII characters as tokens; 128 = pad, 129 = eos."""

    pad_token_id, eos_token_id, unk_token_id = 128, 129, None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text if ord(c) < 128]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 128)

    def convert_tokens_to_ids(self, token):
        return None


@unittest.skipIf(torch is None, "torch/transformers not installed")
class ModelTest(unittest.TestCase):
    def setUp(self):
        from src.clm.agent import CLMHA

        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=130, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                             pad_token_id=128, eos_token_id=129, bos_token_id=1)
        self.model = LlamaForCausalLM(config)
        self.tok = CharTokenizer()
        self.agent = CLMHA(self.model, self.tok, TEMPLATE, EDIT, edit=True, max_new_tokens=6, max_total_tokens=512)

    def test_append_mode_scores_like_h2(self):
        from src.ha.prompts import ANSWER_PREFIX, candidate_ids, chat_prompt, encode
        from src.ha.scoring import candidate_logprobs, softmax

        self.agent.edit = False
        state = CLMState.start("pain")
        state.observe("CT: x\n")
        [record] = self.agent.turn([state])
        message = ha_user_message(TEMPLATE, "pain", {"CT": "x\n"}, ["CT"])
        ids = encode(self.tok, chat_prompt(self.tok, message)) + encode(self.tok, ANSWER_PREFIX)
        expected = softmax(candidate_logprobs(self.model.eval(), ids, candidate_ids(self.tok)))
        self.assertEqual(record.score_prompt_ids, ids)
        for got, want in zip(record.probs, expected):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(record.action, "append_only")

    def test_generate_and_rewrite(self):
        self.model.train()
        completions = self.agent.generate([[60, 61, 62], [60]], sample=False)
        self.assertEqual(len(completions), 2)
        self.assertTrue(all(len(c) <= 6 for c in completions))
        self.assertTrue(self.model.training)  # restored

        text = "<rewrite>n1</rewrite>"
        self.agent.generate = lambda prompts, sample: [[ord(c) for c in text] + [129] for _ in prompts]
        states = [CLMState.start("pain"), CLMState.start("ache")]
        records = self.agent.turn(states, sample=True)
        self.assertEqual([s.context for s in states], ["n1", "n1"])
        self.assertEqual(records[0].action, "rewrite")
        self.assertEqual(records[0].completion_ids[-1], 129)
        self.assertEqual(records[0].score_prompt_ids, self.agent.score_prompt_ids("n1"))

    def test_run_sequences_ragged(self):
        from src.clm.replay import run_sequences

        self.agent.edit = False
        items = [{"history": "a", "blocks": ["CT: 1\n"]}, {"history": "b", "blocks": []},
                 {"history": "c", "blocks": ["CT: 2\n", "MRI: 3\n"]}]
        results = run_sequences(self.agent, items, batch_size=2)
        self.assertEqual([len(r["turns"]) for r in results], [2, 1, 3])
        self.assertEqual(results[2]["context"], "Patient History: c\n\nCT: 2\n\n\nMRI: 3\n")

    def test_update_changes_weights_and_reports(self):
        from src.clm.learning import ha_update

        text = "<rewrite>n</rewrite>"
        self.agent.generate = lambda prompts, sample: [[ord(c) for c in text] + [129] for _ in prompts]
        from src.clm.replay import run_sequences

        item = {"history": "pain", "blocks": ["CT: x\n"]}
        rollouts = run_sequences(self.agent, [item] * 4, sample=True, batch_size=4)
        group = [{"label": "pancreatitis", "turns": r["turns"]} for r in rollouts]
        for k, rollout in enumerate(group):  # one success, three failures: non-zero advantages
            rollout["turns"][-1].probs = [0.0, 0.0, 0.0, 1.0] if k == 0 else [1.0, 0.0, 0.0, 0.0]
        before = [p.detach().clone() for p in self.agent.trainable_parameters()]
        optimizer = torch.optim.AdamW(self.agent.trainable_parameters(), lr=1e-2)
        replay = [(self.agent.score_prompt_ids("CT: y"), "appendicitis")]
        stats = ha_update(self.agent, [group], replay, optimizer, w_eff=0.5)
        self.assertGreater(stats["rl_items"], 0)
        self.assertEqual(stats["sft_rows"], 8 + 1)   # 4 rollouts x 2 turns + 1 replay row
        self.assertAlmostEqual(stats["sft_weight"], 1 / 8)
        self.assertEqual(stats["rewrite_rate"], 1.0)
        changed = any(not torch.equal(a, b) for a, b in zip(before, self.agent.trainable_parameters()))
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
