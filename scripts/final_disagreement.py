"""A1 (docs/FINAL_ANALYSES.md): DA-HA disagreement and deferral rules on the MIMIC test set.

For every system: the agreement table, the accuracy of "defer" (the HA's final hypothesis is the
diagnosis) and "fallback" (the HA only when the DA gave no diagnosis), each against the DA's own
diagnosis with exact McNemar, Holm across the primary systems for "defer". Then "threshold" (defer when
the HA's final confidence >= t): t is chosen on the system's validation episodes (highest validation
accuracy; ties go to the highest t) and applied once to test. Systems without validation episodes
get no threshold.

Uses Edward's edward/da_ha_disagreement.py (load, analyse, scores, compare, mcnemar_exact). Standard
library only.

    python scripts/final_disagreement.py --out-dir outputs/final_analyses
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("da_ha_disagreement", REPO / "edward" / "da_ha_disagreement.py")
dha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dha)

R = "outputs/cost_sweep/cost_1_attempt2"
J = "outputs/contrastive/joint"
# (name, test episodes folder, validation episodes file or None, primary?)
SYSTEMS = [
    ("baseline", f"{R}/test_step_0600", f"{R}/train/episodes/val_step_0600_episodes.csv", True),
    ("step2", f"{R}/frozen_ha_seed0_step_0600", None, True),
    ("h3a", "outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700", "outputs/h3/h3a_ha0/train/episodes/val_step_0700_episodes.csv", True),
    ("h3b", "outputs/h3/h3b_ha0/frozen_ha_seed0_step_0700", "outputs/h3/h3b_ha0/train/episodes/val_step_0700_episodes.csv", True),
    ("h4b", "outputs/contrastive/combined/h2", None, True),
    ("h4c_joint", f"{J}/joint_h2/test_step_0600", None, True),  # original val episodes were overwritten
    ("h4c_frozen", f"{J}/frozen_h2/test_step_0400", f"{J}/frozen_h2/train/episodes/val_step_0400_episodes.csv", True),
    # Secondary (reported, not in the Holm family): run when their test folders exist.
    ("a4b_frozen_step200", f"{J}/frozen_h2/test_step_0200", f"{J}/frozen_h2/train/episodes/val_step_0200_episodes.csv", False),
    ("ext_frozen_step1100", f"{J}/frozen_h2/test_step_1100", f"{J}/frozen_h2/train/episodes/val_step_1100_episodes.csv", False),
    ("a4a_baseline_hidden", f"{R}/hidden_ha_seed0_step_0600", None, False),
    ("a4a_h3a_hidden", "outputs/h3/h3a_ha0/hidden_ha_seed0_step_0700", None, False),
]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def threshold_predictions(rows, t):
    return [h if (c is not None and c >= t and h) else d for _, d, h, c, _ in rows]


def choose_threshold(val_rows) -> tuple[int, float]:
    labels = [r[0] for r in val_rows]
    best = None
    for t in range(0, 11):
        acc = dha.scores(labels, threshold_predictions(val_rows, t))["accuracy"]
        if best is None or acc >= best[1]:  # ">=": ties go to the higher t
            best = (t, acc)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_analyses"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results, missing = {}, []
    for name, test_dir, val_file, primary in SYSTEMS:
        test_file = Path(test_dir) / "test_episodes.csv"
        if not test_file.is_file():
            missing.append(f"{name} ({test_file})")
            continue
        rows = dha.load(test_file)
        res = dha.analyse(rows)
        res.update({"primary": primary, "test_file": str(test_file)})
        labels, da = [r[0] for r in rows], [r[1] for r in rows]
        if val_file and Path(val_file).is_file():
            t, val_acc = choose_threshold(dha.load(val_file))
            pred = threshold_predictions(rows, t)
            res["rule_threshold"] = {"t": t, "val_accuracy": val_acc, "val_file": val_file,
                                     **dha.scores(labels, pred), **dha.compare(labels, da, pred)}
        elif val_file:
            missing.append(f"{name} validation ({val_file})")
        results[name] = res

    primary_p = {n: r["rule_defer"]["mcnemar_p"] for n, r in results.items() if r["primary"]}
    for name, p in holm(primary_p).items():
        results[name]["rule_defer"]["p_holm"] = round(p, 4)

    (args.out_dir / "a1_disagreement.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def cell(r, key):
        return r["cells"].get(key, 0)

    lines = ["# A1: DA-HA disagreement and deferral rules (MIMIC test)", "",
             "Accuracies in %. Wins/losses: patients the rule gets right and the DA wrong / the reverse. "
             "p: exact McNemar; Holm over the primary systems for 'defer'.", "",
             "| System | n | DA | HA final | overrode correct HA | corrected wrong HA | no DA dx, HA right "
             "| defer (wins/losses, p, p Holm) | fallback (p) | threshold t (val acc) -> test (p) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        d, f = r["rule_defer"], r["rule_fallback"]
        th = r.get("rule_threshold")
        th_text = (f"t={th['t']} ({th['val_accuracy']}) -> {th['accuracy']} "
                   f"(+{th['rule_right_da_wrong']}/-{th['da_right_rule_wrong']}, p={th['mcnemar_p']})") if th else "-"
        holm_text = f", Holm {d['p_holm']}" if "p_holm" in d else " (secondary)"
        lines.append(
            f"| {name} | {r['n']} | {r['da']['accuracy']} | {r['ha_final']['accuracy']} | "
            f"{cell(r, 'DA overrode a correct HA')} | {cell(r, 'DA corrected a wrong HA')} | "
            f"{cell(r, 'no DA diagnosis, HA right')} | {d['accuracy']} (+{d['rule_right_da_wrong']}/-{d['da_right_rule_wrong']}, "
            f"p={d['mcnemar_p']}{holm_text}) | {f['accuracy']} (p={f['mcnemar_p']}) | {th_text} |")
    if missing:
        lines += ["", "Not found (skipped): " + "; ".join(missing)]
    (args.out_dir / "a1_disagreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
