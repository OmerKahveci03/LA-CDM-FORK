"""Live integration of TRACE-CDM into ``Environment.run``.

``Environment`` calls three hooks when ``environment.trace`` is set:

- ``start(trajectories, inputs)`` once per ``run()`` call;
- ``before_decision(active_trajectories, step)`` after the Hypothesis Agent and
  before the Decision Agent generates. Shadow arms only record; interventional
  arms append a short audit/critic note to ``trajectory.text`` so the Decision
  Agent reads it as part of its context, like the hypothesis line;
- ``after_step(active_trajectories, step)`` after the environment applied the
  action, to record what the Decision Agent did;
- ``finish(trajectories)`` at the end of ``run()``.

The controller never chooses actions, never returns test results, and only
recommends tests the episode can still request. Results always come from the
dataset oracle in ``Trajectory.get_test_result``.
"""

import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

from src.trace_cdm.audit import build_step_state, deterministic_audit, source_map
from src.trace_cdm.critic import build_prompt, parse_critic, prompt_hash, validate_recommendation
from src.trace_cdm.gate import gate
from src.trace_cdm.schemas import SEVERITY_CRITICAL, SEVERITY_WARNING, AuditResult, CriticResult

ARMS = ("shadow", "shadow_adaptive", "deterministic", "adaptive", "always", "shuffled")


def _sanitize(text: str) -> str:
    """Keep injected notes on one line and free of tokens the environment parses."""
    text = " ".join(text.replace("{", "(").replace("}", ")").split())
    for keyword in ("Observation:", "Action Input:", "Action:", "Thought:"):
        text = text.replace(keyword, keyword[:-1] + " -")
    return text


class TraceController:
    def __init__(
        self,
        arm: str,
        inject: bool,
        critic_mode: str,
        test_list: list[str],
        sources: dict[str, str],
        test_costs_usd: dict[str, float] | None = None,
        critic=None,
        shuffle_critic: bool = False,
        audit_cfg: dict | None = None,
        gate_cfg: dict | None = None,
        max_flags: int = 3,
        max_inject_chars: int = 600,
        max_result_chars: int = 1200,
        redact_text: bool = True,
        seed: int = 0,
    ):
        if critic_mode not in ("none", "adaptive", "always"):
            raise ValueError(f"Unknown critic_mode {critic_mode!r}")
        if critic_mode != "none" and critic is None:
            raise ValueError(f"Arm {arm!r} uses the critic but no critic was loaded")
        self.arm = arm
        self.inject = inject
        self.critic_mode = critic_mode
        self.critic = critic
        self.shuffle_critic = shuffle_critic
        self.test_list = list(test_list)
        self.sources = sources
        self.test_costs_usd = {k.lower(): v for k, v in (test_costs_usd or {}).items()}
        self.audit_cfg = audit_cfg or {}
        self.gate_cfg = gate_cfg or {}
        self.max_flags = max_flags
        self.max_inject_chars = max_inject_chars
        self.max_result_chars = max_result_chars
        self.redact_text = redact_text
        self._rng = random.Random(seed)
        self._records: dict[int, dict] = {}
        self._donor_pool: list[tuple[str, CriticResult]] = []
        self.episodes: list[dict] = []
        self.critic_calls: list[dict] = []
        self.parse_failures = 0
        self.invalid_recommendations = 0
        self.wall_seconds = 0.0

    # ------------------------------------------------------------------ hooks
    def start(self, trajectories, inputs) -> None:
        self._records = {}
        for trajectory, inp in zip(trajectories, inputs):
            self._records[id(trajectory)] = {
                "case_id": str(inp["patient_id"]),
                # Evaluation-only; never placed in step state or critic prompts.
                "ground_truth": inp.get("condition"),
                "steps": [],
                "critic_calls": 0,
                "seen_flags": set(),
                "injected_keys": set(),
            }

    def before_decision(self, trajectories, step: int) -> None:
        t0 = time.perf_counter()
        pending = []  # (trajectory, record, step_row, state, audit)
        for trajectory in trajectories:
            record = self._records[id(trajectory)]
            state = build_step_state(trajectory, step, record["case_id"], self.sources, self.test_list)
            audit = deterministic_audit(state, self.audit_cfg)
            new_flags = [f for f in audit.flags if (f.type, f.item) not in record["seen_flags"]]
            record["seen_flags"].update((f.type, f.item) for f in audit.flags)
            decision = gate(state, AuditResult(new_flags), self.critic_mode, record["critic_calls"], self.gate_cfg)
            row = {
                "step": step,
                "hypothesis": state.hypothesis,
                "confidence": state.confidence,
                "evidence": [
                    {"name": e.name, "source": e.source, "available": e.available, "chars": len(e.text),
                     **({} if self.redact_text else {"text": e.text})}
                    for e in state.evidence
                ],
                "audit": audit.to_dict(),
                "gate": {"invoke": decision.invoke, "reasons": decision.reasons},
                "critic": None,
                "injected": None,
                "action": None,
                "action_input": None,
                "outcome": None,
            }
            if not self.redact_text:
                row["patient_history"] = state.patient_history
            record["steps"].append(row)
            if decision.invoke:
                record["critic_calls"] += 1
            pending.append((trajectory, record, row, state, audit, decision.invoke))

        critic_results = self._run_critic([p for p in pending if p[5]])
        for trajectory, record, row, state, audit, invoked in pending:
            critic_result = critic_results.get(id(trajectory))
            if critic_result is not None:
                row["critic"] = critic_result.to_dict()
            if self.inject:
                note = self._format_note(record, audit, critic_result)
                if note:
                    row["injected"] = note
                    trajectory.text += note
        self.wall_seconds += time.perf_counter() - t0

    def after_step(self, trajectories, step: int) -> None:
        for trajectory in trajectories:
            record = self._records[id(trajectory)]
            log = getattr(trajectory, "action_log", [])
            if not record["steps"] or not log or log[-1]["step"] != step:
                continue
            row = record["steps"][-1]
            row["action"], row["action_input"], row["outcome"] = log[-1]["action"], log[-1]["action_input"], log[-1]["outcome"]

    def finish(self, trajectories) -> None:
        for trajectory in trajectories:
            record = self._records.pop(id(trajectory))
            provided = list(trajectory.provided_tests)
            self.episodes.append({
                "case_id": record["case_id"],
                "arm": self.arm,
                "ground_truth": record["ground_truth"],
                "diagnosis": trajectory.diagnosis,
                "invalid": trajectory.invalid,
                "requested_tests": list(trajectory.requested_tests),
                "provided_tests": provided,
                "num_tests": len(provided),
                "cost_usd": sum(self.test_costs_usd.get(t.lower(), 0) for t in provided),
                "critic_calls": record["critic_calls"],
                "steps": record["steps"],
            })

    # --------------------------------------------------------------- internals
    def _run_critic(self, invoked) -> dict[int, CriticResult]:
        if not invoked:
            return {}
        prompts = [build_prompt(state, audit, self.max_result_chars) for _, _, _, state, audit, _ in invoked]
        outputs = self.critic.generate(prompts)
        results = {}
        real_results = []
        for (trajectory, record, row, state, _, _), prompt, (text, info) in zip(invoked, prompts, outputs):
            parsed = parse_critic(text, state.remaining_tests)
            if parsed.parse_error:
                self.parse_failures += 1
            if parsed.invalid_recommendation:
                self.invalid_recommendations += 1
            call = {"case_id": record["case_id"], "step": state.step, "prompt_sha": prompt_hash(prompt),
                    "parse_error": parsed.parse_error, **info}
            if not self.redact_text:
                call["prompt"], call["raw_output"] = prompt, text
            self.critic_calls.append(call)
            real_results.append((record["case_id"], parsed))
            # Negative control: serve another patient's critique; compute is identical.
            results[id(trajectory)] = (self._shuffled(record["case_id"], state.remaining_tests)
                                       if self.shuffle_critic else parsed)
        if self.shuffle_critic:
            # Real outputs join the donor pool only after this batch is served.
            self._donor_pool.extend((cid, r) for cid, r in real_results if not r.parse_error)
        return results

    def _shuffled(self, case_id: str, remaining_tests: list[str]) -> CriticResult:
        donors = [(cid, r) for cid, r in self._donor_pool if cid != case_id]
        if not donors:
            return CriticResult(shuffled_from="none")
        donor_id, donor = self._rng.choice(donors)
        result = copy.deepcopy(donor)
        result.shuffled_from = donor_id
        result.invalid_recommendation = False
        return validate_recommendation(result, remaining_tests)

    def _format_note(self, record: dict, audit, critic_result: CriticResult | None) -> str:
        parts = []
        new_flags = []
        for flag in audit.of_severity(SEVERITY_CRITICAL, SEVERITY_WARNING):
            key = (flag.type, flag.item)
            if key not in record["injected_keys"]:
                new_flags.append(flag)
        # Critical first, then warnings; cap to keep the context short.
        new_flags.sort(key=lambda f: f.severity != SEVERITY_CRITICAL)
        new_flags = new_flags[: self.max_flags]
        for flag in new_flags:
            record["injected_keys"].add((flag.type, flag.item))
        if new_flags:
            parts.append("Evidence audit: " + "; ".join(f.message for f in new_flags) + ".")

        if critic_result is not None and not critic_result.parse_error and critic_result.shuffled_from != "none":
            items = critic_result.contradictions[:2] + critic_result.evidence_concerns[:2] + critic_result.unsupported_claims[:1]
            text = "Evidence critic:"
            if items:
                text += " " + "; ".join(items) + "."
            if critic_result.recommended_next_step == "request_test":
                text += f" Suggested next test: {critic_result.recommended_test}."
            elif critic_result.recommended_next_step == "proceed":
                text += " The evidence appears sufficient for the current plan."
            if critic_result.reason:
                text += f" ({critic_result.reason})"
            if text != "Evidence critic:":
                parts.append(text)

        if not parts:
            return ""
        note = _sanitize(" ".join(parts))
        if len(note) > self.max_inject_chars:
            note = note[: self.max_inject_chars - 3].rstrip() + "..."
        return note + "\n"

    # ----------------------------------------------------------------- output
    def summary(self) -> dict:
        steps = [s for e in self.episodes for s in e["steps"]]
        n_steps = len(steps)
        flag_counts = Counter(f["type"] for s in steps for f in s["audit"]["flags"])
        gate_reasons = Counter(r for s in steps for r in s["gate"]["reasons"])
        fresh_calls = [c for c in self.critic_calls if not c.get("cached")]
        summary = {
            "arm": self.arm,
            "inject": self.inject,
            "critic_mode": self.critic_mode,
            "shuffle_critic": self.shuffle_critic,
            "episodes": len(self.episodes),
            "decision_points": n_steps,
            "steps_with_warning_or_critical_flags": sum(
                any(f["severity"] in (SEVERITY_CRITICAL, SEVERITY_WARNING) for f in s["audit"]["flags"]) for s in steps),
            "flag_counts": dict(flag_counts),
            "gate_reason_counts": dict(gate_reasons),
            "critic_calls": len(self.critic_calls),
            "critic_call_rate": len(self.critic_calls) / n_steps if n_steps else 0.0,
            "critic_cache_hits": len(self.critic_calls) - len(fresh_calls),
            "critic_parse_failures": self.parse_failures,
            "critic_invalid_recommendations": self.invalid_recommendations,
            "critic_latency_seconds": sum(c.get("latency_seconds", 0.0) for c in self.critic_calls),
            "critic_prompt_tokens": sum(c.get("prompt_tokens", 0) for c in self.critic_calls),
            "critic_completion_tokens": sum(c.get("completion_tokens", 0) for c in self.critic_calls),
            "injected_notes": sum(s["injected"] is not None for s in steps),
            "trace_overhead_seconds": self.wall_seconds,
        }
        if self.critic is not None and hasattr(self.critic, "resource_stats"):
            summary["critic_resources"] = self.critic.resource_stats()
        return summary

    def write(self, output_dir) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "trace_episodes.jsonl").open("w") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode) + "\n")
        with (out / "trace_critic_calls.jsonl").open("w") as f:
            for call in self.critic_calls:
                f.write(json.dumps(call) + "\n")
        summary = self.summary()
        (out / "trace_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def build_trace_controller(trace_cfg: dict, dataset_cfg: dict, test_costs_usd: dict, seed: int):
    """Create the controller for the configured arm, or ``None`` for the native baseline."""
    if not trace_cfg.get("enabled", False):
        return None
    arm = trace_cfg["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown TRACE arm {arm!r}; expected one of {ARMS}")
    critic_cfg = trace_cfg.get("critic") or {}
    critic_mode = critic_cfg.get("mode", "none")
    critic = None
    if critic_mode != "none":
        from src.trace_cdm.critic import load_critic
        critic = load_critic(critic_cfg)
    test_list = list(dataset_cfg["lab_tests"]) + list(dataset_cfg["imaging_tests"]) + list(dataset_cfg["other_tests"])
    return TraceController(
        arm=arm,
        inject=trace_cfg.get("inject", False),
        critic_mode=critic_mode,
        critic=critic,
        shuffle_critic=critic_cfg.get("shuffle", False),
        test_list=test_list,
        sources=source_map(dataset_cfg["lab_tests"], dataset_cfg["imaging_tests"], dataset_cfg["other_tests"]),
        test_costs_usd=test_costs_usd,
        audit_cfg=trace_cfg.get("audit") or {},
        gate_cfg=trace_cfg.get("gate") or {},
        max_flags=trace_cfg.get("max_flags", 3),
        max_inject_chars=trace_cfg.get("max_inject_chars", 600),
        max_result_chars=critic_cfg.get("max_result_chars", 1200),
        redact_text=trace_cfg.get("redact_text", True),
        seed=seed,
    )
