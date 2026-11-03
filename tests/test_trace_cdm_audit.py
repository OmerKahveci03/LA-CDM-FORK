import unittest
from types import SimpleNamespace

from src.trace_cdm.audit import build_step_state, deterministic_audit, parse_lab_values, source_map
from src.trace_cdm.critic import build_prompt, parse_critic
from src.trace_cdm.gate import gate
from src.trace_cdm.schemas import AuditResult, EvidenceItem, StepState

SOURCES = source_map(["Complete Blood Count", "Comprehensive Metabolic Panel", "Urinalysis"], ["CT", "Ultrasound"],
                       ["Physical Examination"])
TESTS = ["Complete Blood Count", "Comprehensive Metabolic Panel", "Urinalysis", "CT", "Ultrasound",
         "Physical Examination"]


def state(evidence=(), hypothesis="appendicitis", confidence=0.6, previous=None, request_count=None, outcome=None):
    evidence = list(evidence)
    return StepState(case_id="c1", step=len(evidence), patient_history="abdominal pain", evidence=evidence,
                     hypothesis=hypothesis, confidence=confidence, previous_hypothesis=previous,
                     request_count=len(evidence) if request_count is None else request_count,
                     remaining_tests=[t for t in TESTS if t not in {e.name for e in evidence}],
                     previous_outcome=outcome)


def lab(name, text):
    return EvidenceItem(name, "lab", True, text)


def types_of(audit):
    return {f.type for f in audit.flags}


class AuditTest(unittest.TestCase):
    def test_parse_lab_values(self):
        self.assertEqual(parse_lab_values("White Blood Cells: 14.2, Lipase: 30\n"),
                         [("White Blood Cells", "14.2"), ("Lipase", "30")])

    def test_unavailable_result(self):
        audit = deterministic_audit(state([EvidenceItem("Urinalysis", "lab", False, "not available.\n")]))
        self.assertIn("unavailable_result", types_of(audit))

    def test_lab_conflict_across_panels_but_not_agreement(self):
        agree = deterministic_audit(state([lab("Complete Blood Count", "Hemoglobin: 13.1\n"),
                                           lab("Comprehensive Metabolic Panel", "Hemoglobin: 13.1\n")]))
        self.assertNotIn("lab_conflict", types_of(agree))
        conflict = deterministic_audit(state([lab("Complete Blood Count", "Hemoglobin: 13.1\n"),
                                              lab("Comprehensive Metabolic Panel", "Hemoglobin: 9.0\n")]))
        self.assertIn("lab_conflict", types_of(conflict))

    def test_invalid_lab_value(self):
        audit = deterministic_audit(state([lab("Complete Blood Count", "White Blood Cells: ___, Hemoglobin: 13\n")]))
        self.assertIn("invalid_lab_value", types_of(audit))

    def test_ambiguous_imaging(self):
        audit = deterministic_audit(state([EvidenceItem("CT", "imaging", True, "Appendicitis cannot be excluded.")]))
        self.assertIn("ambiguous_report", types_of(audit))
        clear = deterministic_audit(state([EvidenceItem("CT", "imaging", True, "Dilated appendix with fat stranding.")]))
        self.assertNotIn("ambiguous_report", types_of(clear))

    def test_hypothesis_flags(self):
        audit = deterministic_audit(state(confidence=0.9))
        self.assertIn("high_confidence_little_evidence", types_of(audit))
        audit = deterministic_audit(state(hypothesis="pancreatitis", previous="appendicitis", confidence=0.3))
        self.assertTrue({"hypothesis_changed", "low_confidence"} <= types_of(audit))
        audit = deterministic_audit(state(hypothesis="unknown", confidence=None))
        self.assertTrue({"hypothesis_unknown", "confidence_missing"} <= types_of(audit))

    def test_no_hypothesis_agent(self):
        audit = deterministic_audit(state(hypothesis=None, confidence=None))
        self.assertFalse({"hypothesis_unknown", "confidence_missing"} & types_of(audit))

    def test_process_errors(self):
        audit = deterministic_audit(state([lab("Urinalysis", "pH: 6\n")], request_count=2, outcome="unknown_test"))
        self.assertTrue({"repeated_request", "previous_action_error"} <= types_of(audit))

    def test_step_state_uses_only_requested_tests(self):
        trajectory = SimpleNamespace(
            requested_tests=["ct"],
            test_results={"ct": "CT text\n", "ultrasound": "HIDDEN\n"},
            test_results_lower_to_original={"ct": "CT", "ultrasound": "Ultrasound"},
            patient_history="pain", hypothesis=["appendicitis"], confidence=[0.7], request_count=1, action_log=[],
        )
        s = build_step_state(trajectory, 1, "c1", SOURCES, TESTS)
        self.assertEqual([e.name for e in s.evidence], ["CT"])
        self.assertNotIn("CT", s.remaining_tests)
        self.assertNotIn("HIDDEN", build_prompt(s, deterministic_audit(s)))


class GateTest(unittest.TestCase):
    def test_modes(self):
        s = state()
        clean = AuditResult([])
        self.assertFalse(gate(s, clean, "none", 0).invoke)
        self.assertTrue(gate(s, clean, "always", 0).invoke)
        self.assertFalse(gate(s, clean, "adaptive", 0).invoke)

    def test_adaptive_triggers_and_budget(self):
        s = state(confidence=0.9)
        audit = deterministic_audit(s)
        decision = gate(s, audit, "adaptive", 0)
        self.assertTrue(decision.invoke)
        self.assertIn("confidence_evidence_disagreement", decision.reasons)
        capped = gate(s, audit, "adaptive", 4, {"max_calls_per_episode": 4})
        self.assertFalse(capped.invoke)

    def test_low_confidence_needs_evidence(self):
        self.assertFalse(gate(state(confidence=0.2), deterministic_audit(state(confidence=0.2)), "adaptive", 0).invoke)
        s = state([lab("Urinalysis", "pH: 6\n")], confidence=0.2)
        self.assertIn("low_confidence_after_testing", gate(s, deterministic_audit(s), "adaptive", 0).reasons)


class CriticParseTest(unittest.TestCase):
    REMAINING = ["CT", "Ultrasound"]

    def test_valid(self):
        r = parse_critic('Sure: {"evidence_concerns": ["x"], "unsupported_claims": [], "contradictions": [], '
                         '"recommended_next_step": "request_test", "recommended_test": "ct", "reason": "r", '
                         '"evidence_quality": 0.5}', self.REMAINING)
        self.assertIsNone(r.parse_error)
        self.assertEqual(r.recommended_test, "CT")
        self.assertEqual(r.evidence_quality, 0.5)

    def test_malformed_falls_back(self):
        for text in ("nonsense", '{"evidence_concerns": "not a list", "recommended_next_step": "proceed"}',
                     '{"recommended_next_step": "diagnose"}', "[1, 2]"):
            r = parse_critic(text, self.REMAINING)
            self.assertIsNotNone(r.parse_error, text)
            self.assertEqual(r.recommended_next_step, "review")

    def test_unavailable_test_recommendation_dropped(self):
        r = parse_critic('{"recommended_next_step": "request_test", "recommended_test": "MRI"}', self.REMAINING)
        self.assertIsNone(r.recommended_test)
        self.assertEqual(r.recommended_next_step, "review")
        self.assertTrue(r.invalid_recommendation)

    def test_quality_out_of_range_ignored(self):
        r = parse_critic('{"recommended_next_step": "proceed", "evidence_quality": 7}', self.REMAINING)
        self.assertIsNone(r.evidence_quality)


if __name__ == "__main__":
    unittest.main()
