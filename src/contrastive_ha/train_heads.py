"""Stage 1b of H4: train the projection heads on cached embeddings, choose on validation, score test.

    python -m src.contrastive_ha.train_heads --emb-dir outputs/contrastive/embeddings/base \\
        --out-dir outputs/contrastive/heads/base --scores-dir outputs/contrastive/scores/base

Fast (the backbone is frozen and its embeddings are cached): minutes on one GPU or a CPU.

1. Train on ``train.pt`` (the H2 training views) with InfoNCE over the four diseases.
2. After every epoch, evaluate on ``val.pt``; keep the epoch with the best validation accuracy
   (ties: lower validation loss). The test records are not looked at until this choice is made.
3. Save ``heads.pt`` (with the action embeddings) and ``curve.csv``, then score every
   ``test/<arm>.pt``: ``<scores-dir>/<arm>/{test_episodes.csv, ha_probs.csv, summary.json}``,
   in the same format as the H2 scores, so scripts/test_stats.py compares them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.contrastive_ha.heads import ContrastiveHeads, save_heads
from src.ha.prompts import DISEASES
from src.ha.records import probability_summary, scored_row


def load(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu")
    blob["x"] = blob["x"].float()
    return blob


@torch.no_grad()
def evaluate(heads: ContrastiveHeads, x: torch.Tensor, y: torch.Tensor, actions: torch.Tensor) -> dict:
    heads.eval()
    logits = heads.logits(x, actions)
    loss = torch.nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(-1) == y).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load(args.emb_dir / "train.pt"), load(args.emb_dir / "val.pt")
    actions = torch.load(args.emb_dir / "actions.pt", map_location="cpu")["x"].float()
    xt, yt, xv, yv, a = train["x"].to(device), train["y"].to(device), val["x"].to(device), val["y"].to(device), actions.to(device)

    heads = ContrastiveHeads(xt.size(1), args.width, args.depth, args.out_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n_params = sum(p.numel() for p in heads.parameters())
    print(f"[heads] {args.emb_dir}: train {len(yt)}, val {len(yv)}, {n_params:,} head parameters, device {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curve, best, best_state = [], None, None
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        heads.train()
        order = torch.randperm(len(yt), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start: start + args.batch_size]
            loss = heads.loss(xt[idx], a, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        schedule.step()
        tr, va = evaluate(heads, xt, yt, a), evaluate(heads, xv, yv, a)
        row = {"epoch": epoch, "train_loss": total / len(yt), "train_accuracy": tr["accuracy"],
               "val_loss": va["loss"], "val_accuracy": va["accuracy"],
               "temperature": 1 / heads.logit_scale.exp().item()}
        curve.append(row)
        key = (va["accuracy"], -va["loss"])
        if best is None or key > best[0]:
            best = (key, epoch)
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}
        print(f"[heads] epoch {epoch}: train acc {tr['accuracy']:.3f}  val acc {va['accuracy']:.3f}  "
              f"val loss {va['loss']:.3f}", flush=True)

    with (args.out_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    heads.load_state_dict(best_state)
    meta = {"emb_dir": str(args.emb_dir), "best_epoch": best[1], "val_accuracy": best[0][0],
            "val_loss": -best[0][1], "args": {k: str(v) for k, v in vars(args).items()}}
    save_heads(args.out_dir / "heads.pt", heads, actions, meta)
    (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(f"[heads] chosen epoch {best[1]} (val accuracy {best[0][0]:.3f}) -> {args.out_dir / 'heads.pt'}")

    # Test scoring, only after the choice above.
    from src.utils.episode_log import read_rows, write_rows

    def first_episodes(path: Path) -> list[dict]:  # as src/score_ha.py, without its model imports
        seen, rows = set(), []
        for row in read_rows(path):
            if row["patient_id"] not in seen:
                seen.add(row["patient_id"])
                rows.append(row)
        return rows

    overview = {}
    for path in sorted((args.emb_dir / "test").glob("*.pt")):
        arm = path.stem
        blob = load(path)
        probs = heads.probs(blob["x"].to(device), a).cpu().tolist()
        sources = {str(s["patient_id"]): s for s in first_episodes(Path(blob["source"]))}
        rows = [scored_row(sources[str(pid)], p) for pid, p in zip(blob["patient_id"], probs)]
        arm_dir = args.scores_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_rows(arm_dir / "test_episodes.csv", rows)
        with (arm_dir / "ha_probs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "label", "prompt_tokens", *(f"p_{d}" for d in DISEASES)])
            for pid, row, tokens, p in zip(blob["patient_id"], rows, blob["prompt_tokens"], probs):
                writer.writerow([pid, row["label"], tokens, *p])
        summary = probability_summary([r["label"] for r in rows], probs) | {"arm": arm, "heads": str(args.out_dir)}
        (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        overview[arm] = summary
        print(f"[heads] test {arm}: accuracy {summary['accuracy']:.3f}, AUROC {summary['auroc_ovr_exact']:.3f}, "
              f"ECE {summary['ece_10_bins']:.3f}, n={summary['n']}")
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    (args.scores_dir / "overview.json").write_text(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
