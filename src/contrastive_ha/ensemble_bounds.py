"""A2 (docs/FINAL_ANALYSES.md): confidence bounds for the CLM HA from an ensemble of heads.

K sets of heads are trained on the cached embeddings of the chosen backbone, each with its own seed
and a bootstrap resample of the *training patients* (all of a patient's views together), and each
picks its epoch on the full validation set as in stage 1. Per record, the mean probability and the
standard deviation over members give bounds LCB = mean - 2 sd and UCB = mean + 2 sd (factor fixed in
advance). A record is "decisive" when LCB(top) > UCB(runner-up).

Reported on the H4a record sets (primary: A_da2, the baseline DA's tests):
  1. how well the bound margin flags wrong answers (AUROC), against the plain mean confidence and a
     single member's confidence, with a bootstrap CI of the difference;
  2. accuracy of decisive and non-decisive records;
  3. "diagnose or test more" offline: non-decisive A_da2 records take the ensemble's all-tests (D_all)
     answer instead; accuracy and extra test cost against never and always using all tests;
  4. ensemble mean against a single member (accuracy, ECE).

    python -m src.contrastive_ha.ensemble_bounds --emb-dir outputs/contrastive/embeddings/h2 \\
        --out-dir outputs/final_analyses
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch

from src.contrastive_ha.heads import ContrastiveHeads
from src.ha.prompts import DISEASES
from src.ha.records import probability_summary
from src.utils.episode_log import read_rows

Z = 2.0  # bound width in standard deviations, fixed in docs/FINAL_ANALYSES.md


def load(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu")
    blob["x"] = blob["x"].float()
    return blob


@torch.no_grad()
def val_metrics(heads, x, y, a):
    heads.eval()
    logits = heads.logits(x, a)
    return (logits.argmax(-1) == y).float().mean().item(), torch.nn.functional.cross_entropy(logits, y).item()


def train_member(k: int, train: dict, val: dict, a: torch.Tensor, args, device) -> tuple[ContrastiveHeads, dict]:
    torch.manual_seed(k)
    rng = random.Random(k)
    by_patient: dict[str, list[int]] = {}
    for i, pid in enumerate(train["patient_id"]):
        by_patient.setdefault(str(pid), []).append(i)
    patients = sorted(by_patient)
    rows = [i for _ in patients for i in by_patient[rng.choice(patients)]]  # bootstrap over patients
    idx_all = torch.tensor(rows)
    xt, yt = train["x"].to(device), train["y"].to(device)
    xv, yv = val["x"].to(device), val["y"].to(device)
    heads = ContrastiveHeads(xt.size(1), args.width, args.depth, args.out_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(k)
    best = None
    for epoch in range(1, args.epochs + 1):
        heads.train()
        order = idx_all[torch.randperm(len(idx_all), generator=generator)].to(device)
        for start in range(0, len(order), args.batch_size):
            b = order[start: start + args.batch_size]
            loss = heads.loss(xt[b], a, yt[b])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        schedule.step()
        acc, vloss = val_metrics(heads, xv, yv, a)
        if best is None or (acc, -vloss) > best[0]:
            best = ((acc, -vloss), epoch, {n: v.detach().clone() for n, v in heads.state_dict().items()})
    heads.load_state_dict(best[2])
    info = {"member": k, "epoch": best[1], "val_accuracy": best[0][0], "val_loss": -best[0][1],
            "train_rows": len(rows), "unique_patients": len(set(str(train["patient_id"][i]) for i in rows))}
    print(f"[ensemble] member {k}: epoch {info['epoch']}, val acc {info['val_accuracy']:.3f}, "
          f"{info['unique_patients']} unique patients", flush=True)
    return heads, info


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    """Probability that a positive scores above a negative (ties count half)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    ranked = sorted((s, i) for i, s in enumerate(scores))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for m in range(i, j + 1):
            ranks[ranked[m][1]] = (i + j) / 2 + 1
        i = j + 1
    pos_rank = sum(r for r, p in zip(ranks, positives) if p)
    return (pos_rank - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def analyse(member_probs: torch.Tensor, y: torch.Tensor, resamples: int, seed: int) -> dict:
    """member_probs: [K, N, 4]."""
    mean, sd = member_probs.mean(0), member_probs.std(0, unbiased=True)
    order = mean.argsort(-1, descending=True)
    top, second = order[:, 0], order[:, 1]
    rows = torch.arange(len(y))
    p_top, sd_top = mean[rows, top], sd[rows, top]
    p_sec, sd_sec = mean[rows, second], sd[rows, second]
    margin = (p_top - Z * sd_top) - (p_sec + Z * sd_sec)
    decisive = margin > 0
    correct = top == y
    single = member_probs[0]
    single_correct = single.argmax(-1) == y
    wrong = [not c for c in correct.tolist()]
    m_scores, p_scores = (-margin).tolist(), (-p_top).tolist()
    s_scores = (-single.max(-1).values).tolist()
    auc_margin, auc_mean, auc_single = auroc(m_scores, wrong), auroc(p_scores, wrong), auroc(s_scores, wrong)
    rng = random.Random(seed)
    diffs = []
    n = len(wrong)
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        w = [wrong[i] for i in idx]
        a1, a2 = auroc([m_scores[i] for i in idx], w), auroc([p_scores[i] for i in idx], w)
        if a1 is not None and a2 is not None:
            diffs.append(a1 - a2)
    diffs.sort()
    labels = [DISEASES[i] for i in y.tolist()]
    ens_summary = probability_summary(labels, mean.tolist())
    single_summary = probability_summary(labels, single.tolist())
    dec, nondec = decisive.tolist(), [not d for d in decisive.tolist()]
    acc = lambda mask: (sum(c for c, m in zip(correct.tolist(), mask) if m) / max(1, sum(mask))) if any(mask) else None
    return {
        "n": n,
        "ensemble_accuracy": ens_summary["accuracy"], "ensemble_ece": ens_summary["ece_10_bins"],
        "single_accuracy": single_summary["accuracy"], "single_ece": single_summary["ece_10_bins"],
        "ensemble_vs_single": {"wins": int(((correct) & (~single_correct)).sum()),
                               "losses": int(((~correct) & single_correct).sum())},
        "mean_sd_top": sd_top.mean().item(),
        "decisive_share": sum(dec) / n, "accuracy_decisive": acc(dec), "accuracy_nondecisive": acc(nondec),
        "auroc_wrong": {"margin": auc_margin, "mean_confidence": auc_mean, "single_confidence": auc_single,
                        "margin_minus_mean": (auc_margin - auc_mean) if auc_margin is not None and auc_mean is not None else None,
                        "diff_ci95": [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if diffs else None},
        "_decisive": dec, "_top": top.tolist(), "_correct": correct.tolist(),
    }


def first_costs(source: str) -> dict[str, float]:
    out = {}
    for row in read_rows(Path(source)):
        out.setdefault(str(row["patient_id"]), float(row["cost_usd"] or 0.0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb-dir", type=Path, default=Path("outputs/contrastive/embeddings/h2"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_analyses"))
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = load(args.emb_dir / "train.pt"), load(args.emb_dir / "val.pt")
    a = torch.load(args.emb_dir / "actions.pt", map_location="cpu")["x"].float().to(device)
    sets = {"val": val} | {p.stem: load(p) for p in sorted((args.emb_dir / "test").glob("*.pt"))}
    print(f"[ensemble] {args.members} members on {args.emb_dir} ({device}); record sets: {', '.join(sets)}", flush=True)

    members, probs = [], {name: [] for name in sets}
    for k in range(args.members):
        heads, info = train_member(k, train, val, a, args, device)
        members.append(info)
        for name, blob in sets.items():
            probs[name].append(heads.probs(blob["x"].to(device), a).cpu())
    stacked = {name: torch.stack(p) for name, p in probs.items()}  # [K, N, 4]

    results = {"members": members, "z": Z, "record_sets": {}}
    for name, blob in sets.items():
        res = analyse(stacked[name], blob["y"], args.resamples, seed=0)
        results["record_sets"][name] = {k: v for k, v in res.items() if not k.startswith("_")}
        res_full = res
        with (out / f"a2_ensemble_probs_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "label", *(f"mean_{d}" for d in DISEASES), *(f"sd_{d}" for d in DISEASES), "decisive"])
            mean, sd = stacked[name].mean(0), stacked[name].std(0, unbiased=True)
            for i, pid in enumerate(blob["patient_id"]):
                writer.writerow([pid, DISEASES[int(blob["y"][i])], *[round(v, 5) for v in mean[i].tolist()],
                                 *[round(v, 5) for v in sd[i].tolist()], int(res_full["_decisive"][i])])
        if name == "A_da2":
            a_res = res_full
        if name == "D_all":
            d_res = res_full

    # 3. "Diagnose or test more" on A_da2, with the all-tests answer for non-decisive records.
    if "A_da2" in sets and "D_all" in sets:
        pa, pd = [str(p) for p in sets["A_da2"]["patient_id"]], [str(p) for p in sets["D_all"]["patient_id"]]
        d_index = {pid: i for i, pid in enumerate(pd)}
        cost_a, cost_d = first_costs(sets["A_da2"]["source"]), first_costs(sets["D_all"]["source"])
        never, hybrid, always, c_never, c_hybrid, c_always = [], [], [], [], [], []
        for i, pid in enumerate(pa):
            j = d_index[pid]
            never.append(a_res["_correct"][i])
            always.append(d_res["_correct"][j])
            take_all = not a_res["_decisive"][i]
            hybrid.append(d_res["_correct"][j] if take_all else a_res["_correct"][i])
            c_never.append(cost_a[pid])
            c_always.append(cost_d[pid])
            c_hybrid.append(cost_d[pid] if take_all else cost_a[pid])
        n = len(pa)
        wins = sum(h and not v for h, v in zip(hybrid, never))
        losses = sum(v and not h for h, v in zip(hybrid, never))
        results["test_more_policy_A_da2"] = {
            "n": n, "share_tested_more": sum(not d for d in a_res["_decisive"]) / n,
            "accuracy": {"never": sum(never) / n, "hybrid": sum(hybrid) / n, "always_all_tests": sum(always) / n},
            "mean_cost_usd": {"never": sum(c_never) / n, "hybrid": sum(c_hybrid) / n, "always_all_tests": sum(c_always) / n},
            "hybrid_vs_never": {"wins": wins, "losses": losses, "mcnemar_p": mcnemar_exact(wins, losses)},
            "note": "Offline upper-bound sketch: assumes the extra tests would be ordered; not a deployed policy.",
        }

    (out / "a2_ensemble_bounds.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [f"# A2: CLM head ensemble, bounds mean +/- {Z} sd ({args.members} members)", "",
             "| Record set | n | ensemble acc | single acc | ensemble ECE | single ECE | decisive share | acc decisive "
             "| acc non-decisive | AUROC wrong: margin / mean conf / single conf | margin - mean [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    num = lambda v: "-" if v is None else f"{v:.3f}"
    for name, r in results["record_sets"].items():
        au = r["auroc_wrong"]
        ci = au["diff_ci95"]
        lines.append(f"| {name} | {r['n']} | {pct(r['ensemble_accuracy'])} | {pct(r['single_accuracy'])} | "
                     f"{pct(r['ensemble_ece'])} | {pct(r['single_ece'])} | {pct(r['decisive_share'])} | "
                     f"{pct(r['accuracy_decisive'])} | {pct(r['accuracy_nondecisive'])} | "
                     f"{num(au['margin'])} / {num(au['mean_confidence'])} / {num(au['single_confidence'])} | "
                     f"{num(au['margin_minus_mean'])} [{num(ci[0]) if ci else '-'}, {num(ci[1]) if ci else '-'}] |")
    if "test_more_policy_A_da2" in results:
        t = results["test_more_policy_A_da2"]
        lines += ["", "**Diagnose or test more (A_da2, offline sketch):** "
                  f"{100 * t['share_tested_more']:.1f}% of records not decisive. Accuracy never/hybrid/always-all-tests: "
                  f"{100 * t['accuracy']['never']:.1f} / {100 * t['accuracy']['hybrid']:.1f} / {100 * t['accuracy']['always_all_tests']:.1f}%; "
                  f"mean cost ${t['mean_cost_usd']['never']:.0f} / ${t['mean_cost_usd']['hybrid']:.0f} / ${t['mean_cost_usd']['always_all_tests']:.0f}; "
                  f"hybrid vs never +{t['hybrid_vs_never']['wins']}/-{t['hybrid_vs_never']['losses']}, p={t['hybrid_vs_never']['mcnemar_p']:.3f}."]
    lines += ["", "Members: " + ", ".join(f"#{m['member']} epoch {m['epoch']} val {100 * m['val_accuracy']:.1f}" for m in members)]
    (out / "a2_ensemble_bounds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
