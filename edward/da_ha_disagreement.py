"""DA-HA disagreement on a set of test episodes (Omer's NEXT_EXPERIMENTS.md item 1, run on my external set).

For each episode: did the decision agent (DA) give a diagnosis, does it agree with the hypothesis agent's (HA) final
hypothesis, and which of the two is right. Also the accuracy of two rules that need no training:
  - "defer": the diagnosis is the HA's final hypothesis;
  - "fallback": the DA's diagnosis, or the HA's final hypothesis when the DA gave none (invalid format, step limit);
and of "defer when the HA's final confidence >= t" for every t. The sweep is descriptive only: a threshold must be
chosen on validation episodes, and this external set is test-only.

Reads either Omer's LA-CDM-FORK episode file (test_episodes.csv, one row per episode) or the episodes.jsonl of my own
zero-shot harness (lacdm/evaluate.py). Standard library only, so it runs anywhere (also in Omer's env).

  python da_ha_disagreement.py <episodes file> [--name NAME] [--out result.json]

(Copied unchanged from Edward Kemp, /blue/cai6734/kempaedward/LA-CDM-FORK/slurm/ext_v2/, on 2026-11-24; reviewed:
reads one episodes file, writes one JSON. Its confidence thresholds 0-10 match the 0-10 scale our episode files
store, src/environment/frozen_ha_environment.py.)
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DISEASES = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")


def norm(x):
    x = (x or "").strip().lower()
    return x if x in DISEASES else ""


def load(path):
    """Rows of (label, da_dx or "", ha_final or "", ha_conf or None, end_reason)."""
    p = Path(path)
    rows = []
    if p.suffix == ".csv":
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                conf = r.get("final_confidence", "")
                rows.append((norm(r["label"]), norm(r.get("diagnosis")), norm(r.get("final_hypothesis")),
                             float(conf) if conf not in ("", None) else None, r.get("end_reason", "")))
    else:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            hyps = [h for h in r.get("hypotheses") or [] if h.get("hypothesis")]
            last = hyps[-1] if hyps else {}
            rows.append((norm(r["label"]), norm(r.get("pred")), norm(last.get("hypothesis")),
                         last.get("confidence"), r.get("outcome", "")))
    return rows


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def scores(labels, preds):
    n = len(labels)
    acc = sum(p == y for y, p in zip(labels, preds)) / n
    per = {}
    for d in DISEASES:
        idx = [i for i, y in enumerate(labels) if y == d]
        if idx:
            per[d] = sum(preds[i] == d for i in idx) / len(idx)
    return {"accuracy": round(100 * acc, 1), "mean_class_accuracy": round(100 * sum(per.values()) / len(per), 1),
            "per_disease": {d: round(100 * v, 1) for d, v in per.items()}}


def compare(labels, a, b):
    """Paired comparison of rule b against the DA's own diagnosis a: wins, losses, exact McNemar p."""
    wins = sum(pb == y and pa != y for y, pa, pb in zip(labels, a, b))
    losses = sum(pa == y and pb != y for y, pa, pb in zip(labels, a, b))
    return {"rule_right_da_wrong": wins, "da_right_rule_wrong": losses, "mcnemar_p": round(mcnemar_exact(wins, losses), 4)}


def analyse(rows):
    labels = [r[0] for r in rows]
    da = [r[1] for r in rows]
    ha = [r[2] for r in rows]
    conf = [r[3] for r in rows]
    cells = Counter()
    for y, d, h in zip(labels, da, ha):
        if not d:
            cells["no DA diagnosis, HA right" if h == y else "no DA diagnosis, HA wrong"] += 1
        elif d == h:
            cells["agree, both right" if d == y else "agree, both wrong"] += 1
        elif h == y:
            cells["DA overrode a correct HA"] += 1
        elif d == y:
            cells["DA corrected a wrong HA"] += 1
        else:
            cells["disagree, both wrong"] += 1
    defer = [h for h in ha]
    fallback = [d or h for d, h in zip(da, ha)]
    out = {
        "n": len(rows),
        "da_diagnosed": sum(bool(d) for d in da),
        "end_reasons": dict(Counter(r[4] for r in rows)),
        "cells": dict(cells),
        "da": scores(labels, da),
        "ha_final": scores(labels, ha),
        "rule_defer": {**scores(labels, defer), **compare(labels, da, defer)},
        "rule_fallback": {**scores(labels, fallback), **compare(labels, da, fallback)},
    }
    sweep = []
    known = [c for c in conf if c is not None]
    if known:
        for t in range(0, 11):
            pred = [h if (c is not None and c >= t and h) else d for d, h, c in zip(da, ha, conf)]
            sweep.append({"t": t, "deferred": sum(1 for d, h, c in zip(da, ha, conf) if c is not None and c >= t and h),
                          **scores(labels, pred)})
        out["sweep_defer_if_conf_ge_t"] = sweep
        cal = defaultdict(lambda: [0, 0])
        for y, h, c in zip(labels, ha, conf):
            if c is not None:
                cal[int(c)][0] += h == y
                cal[int(c)][1] += 1
        out["ha_calibration"] = {str(k): {"right": v[0], "n": v[1], "accuracy": round(100 * v[0] / v[1], 1)}
                                 for k, v in sorted(cal.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--name", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {"name": a.name or a.episodes, "episodes_file": str(a.episodes), **analyse(load(a.episodes))}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"{res['name']}: n={res['n']}, DA diagnosed {res['da_diagnosed']}")
    for k in ("da", "ha_final", "rule_defer", "rule_fallback"):
        r = res[k]
        extra = f"  (+{r['rule_right_da_wrong']} / -{r['da_right_rule_wrong']}, p={r['mcnemar_p']})" if "mcnemar_p" in r else ""
        print(f"  {k:14s} acc {r['accuracy']:5.1f}  mean class {r['mean_class_accuracy']:5.1f}{extra}")
    for k, v in sorted(res["cells"].items()):
        print(f"  {k:30s} {v}")
    if "ha_calibration" in res:
        print("  HA final confidence -> accuracy: " +
              ", ".join(f"{k}: {v['accuracy']:.0f}% of {v['n']}" for k, v in res["ha_calibration"].items()))


if __name__ == "__main__":
    main()
