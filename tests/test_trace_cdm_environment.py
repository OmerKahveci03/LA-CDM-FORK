"""Integration tests: TRACE hooks inside the real Environment.run loop.

A scripted fake vLLM model stands in for Llama so the tests run on CPU. If trl is
not installed, a minimal stand-in for ``maybe_apply_chat_template`` is used.
"""

import json
import sys
import types
import unittest
from types import SimpleNamespace

try:
    import trl.data_utils  # noqa: F401
except ImportError:
    stub = types.ModuleType("trl.data_utils")
    stub.maybe_apply_chat_template = lambda example, tokenizer: {
        "prompt": tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)}
    sys.modules.setdefault("trl", types.ModuleType("trl"))
    sys.modules["trl.data_utils"] = stub

from src.environment.environment import Environment  # noqa: E402
from src.trace_cdm.audit import source_map  # noqa: E402
from src.trace_cdm.controller import TraceController  # noqa: E402

LABS = ["Complete Blood Count", "Urinalysis"]
IMAGING = ["CT", "Ultrasound"]
OTHER = ["Physical Examination"]
TESTS = LABS + IMAGING + OTHER
DISEASES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]

DA_TEMPLATE = "Diagnose the patient.\n{patient_history}\n---\n{first_hypothesis}"
HA_TEMPLATE = "Create a hypothesis.\n{provided_test_results}"


class FakeTokenizer:
    eos_token_id = -1
    eos_token = "</s>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [len(w) for w in text.split()]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}


class FakeLLM:
    """Hypothesis Agent always says appendicitis (9/10); the Decision Agent requests
    Ultrasound, then Urinalysis, then diagnoses, regardless of injected notes."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.da_prompts = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outputs = []
        for prompt in prompts:
            if "Create a hypothesis" in prompt:
                text = "Hypothesis: appendicitis\nConfidence: 9"
            else:
                self.da_prompts.append(prompt)
                n_obs = prompt.count("Observation:")
                if n_obs == 0:
                    text = "Thought: image first\nAction: Test\nAction Input: Ultrasound\n"
                elif n_obs == 1:
                    text = "Thought: check urine\nAction: Test\nAction Input: Urinalysis\n"
                else:
                    text = "Thought: enough\nAction: Diagnosis\nAction Input: appendicitis"
            outputs.append(SimpleNamespace(
                prompt=prompt,
                prompt_token_ids=self.tokenizer.encode(prompt),
                outputs=[SimpleNamespace(text=text, token_ids=self.tokenizer.encode(text))],
            ))
        return outputs


class FakeCritic:
    def __init__(self, recommended_test="CT"):
        self.prompts = []
        self.recommended_test = recommended_test

    def generate(self, prompts):
        self.prompts.extend(prompts)
        body = {"evidence_concerns": ["ultrasound is equivocal"], "unsupported_claims": [], "contradictions": [],
                "recommended_next_step": "request_test", "recommended_test": self.recommended_test,
                "reason": "imaging is inconclusive", "evidence_quality": 0.4}
        info = {"cached": False, "latency_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        return [(json.dumps(body), info) for _ in prompts]


def make_inputs():
    results = {
        "Complete Blood Count": "White Blood Cells: 14.2, Hemoglobin: 13.1\n",
        "Urinalysis": "not available.\n",
        "CT": "SECRET_UNREQUESTED_CT_FINDING\n",
        "Ultrasound": "Inflamed tubular structure; perforation cannot be excluded.\n",
        "Physical Examination": "RLQ tenderness.\n",
    }
    return [
        {"patient_id": pid, "patient_history": "Adult with abdominal pain for one day.",
         "condition": "LABEL_SENTINEL", "test_results": dict(results)}
        for pid in ("case-1", "case-2")
    ]


def run_env(controller=None):
    env = Environment(DA_TEMPLATE, max_length=4096, max_steps=13, disease_list=DISEASES, test_list=TESTS,
                      generate_hypothesis=True, hypothesis_prompt_template=HA_TEMPLATE,
                      generate_confidence_calibration=True, seed=1)
    env.trace = controller
    model = FakeLLM()
    inputs = env.construct_prompt(make_inputs())
    prompts_text = [model.tokenizer.apply_chat_template(i["prompt"]) for i in inputs]
    output = env.run(prompts_text, inputs, model, None, model.tokenizer)
    return output, model


def make_controller(arm, inject, critic_mode="none", critic=None, shuffle=False):
    return TraceController(arm=arm, inject=inject, critic_mode=critic_mode, critic=critic, shuffle_critic=shuffle,
                           test_list=TESTS, sources=source_map(LABS, IMAGING, OTHER),
                           test_costs_usd={"ultrasound": 1288, "urinalysis": 50}, seed=0)


class TraceEnvironmentTest(unittest.TestCase):
    def test_shadow_reproduces_baseline_exactly(self):
        baseline, base_model = run_env(None)
        controller = make_controller("shadow", inject=False)
        shadow, shadow_model = run_env(controller)
        self.assertEqual(baseline.completion_ids, shadow.completion_ids)
        self.assertEqual(baseline.completion_mask, shadow.completion_mask)
        self.assertEqual(baseline.reward_function_kwargs, shadow.reward_function_kwargs)
        self.assertEqual(base_model.da_prompts, shadow_model.da_prompts)
        self.assertEqual(sum(s["injected"] is not None for e in controller.episodes for s in e["steps"]), 0)

    def test_episode_export(self):
        controller = make_controller("shadow", inject=False)
        run_env(controller)
        self.assertEqual([e["case_id"] for e in controller.episodes], ["case-1", "case-2"])
        episode = controller.episodes[0]
        self.assertEqual(episode["diagnosis"], "appendicitis")
        self.assertEqual(episode["provided_tests"], ["ultrasound"])
        self.assertEqual(episode["cost_usd"], 1288)
        self.assertEqual(
            [(s["action"], s["action_input"], s["outcome"]) for s in episode["steps"]],
            [("test", "ultrasound", "result"), ("test", "urinalysis", "not_available"),
             ("diagnosis", "appendicitis", "diagnosed")],
        )
        # The label lives outside the steps and redacted exports carry no result text.
        self.assertNotIn("LABEL_SENTINEL", json.dumps(episode["steps"]))
        self.assertNotIn("text", episode["steps"][1]["evidence"][0])
        flag_types = {f["type"] for f in episode["steps"][1]["audit"]["flags"]}
        self.assertIn("ambiguous_report", flag_types)
        self.assertIn("unavailable_result", {f["type"] for f in episode["steps"][2]["audit"]["flags"]})

    def test_deterministic_arm_injects_into_decision_agent_context(self):
        controller = make_controller("deterministic", inject=True)
        _, model = run_env(controller)
        injected = [p for p in model.da_prompts if "Evidence audit:" in p]
        self.assertTrue(injected)
        self.assertTrue(any("hedged findings" in p for p in injected))
        # Notes are injected once per finding, not repeated every step.
        final_prompt = model.da_prompts[-1]
        self.assertEqual(final_prompt.count("Ultrasound contains hedged findings"), 1)

    def test_adaptive_critic_sees_only_current_evidence(self):
        critic = FakeCritic()
        controller = make_controller("adaptive", inject=True, critic_mode="adaptive", critic=critic)
        _, model = run_env(controller)
        self.assertTrue(critic.prompts)
        for prompt in critic.prompts:
            self.assertNotIn("SECRET_UNREQUESTED_CT_FINDING", prompt)
            self.assertNotIn("LABEL_SENTINEL", prompt)
        # Step 0 has no results yet, so no result text can appear in its prompt.
        self.assertNotIn("Inflamed tubular", critic.prompts[0])
        self.assertTrue(any("Suggested next test: CT" in p for p in model.da_prompts))
        summary = controller.summary()
        self.assertGreater(summary["critic_calls"], 0)
        self.assertLess(summary["critic_call_rate"], 1.0)  # gated, not always-on

    def test_always_arm_calls_critic_every_decision(self):
        critic = FakeCritic()
        controller = make_controller("always", inject=True, critic_mode="always", critic=critic)
        run_env(controller)
        self.assertEqual(controller.summary()["critic_call_rate"], 1.0)

    def test_invalid_recommendation_is_not_injected(self):
        critic = FakeCritic(recommended_test="Colonoscopy")
        controller = make_controller("always", inject=True, critic_mode="always", critic=critic)
        _, model = run_env(controller)
        self.assertFalse(any("Colonoscopy" in p for p in model.da_prompts))
        self.assertGreater(controller.invalid_recommendations, 0)

    def test_shuffled_arm_uses_other_patients_critiques(self):
        critic = FakeCritic()
        controller = make_controller("shuffled", inject=True, critic_mode="always", critic=critic, shuffle=True)
        run_env(controller)
        donors = [(e["case_id"], s["critic"]["shuffled_from"]) for e in controller.episodes
                  for s in e["steps"] if s["critic"]]
        self.assertTrue(donors)
        for case_id, donor in donors:
            self.assertNotEqual(case_id, donor)
        self.assertTrue(any(donor not in (None, "none") for _, donor in donors))


if __name__ == "__main__":
    unittest.main()
