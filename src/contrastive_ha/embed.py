"""Stage 1a of H4: embed every record once with the frozen backbone.

    python -m src.contrastive_ha.embed --backbone base --out-dir outputs/contrastive/embeddings/base \\
        --arm A_da2=outputs/cost_sweep/cost_1_attempt2/test_step_0600/test_episodes.csv ...

Records (the same as H2):

- ``train.pt``: every training patient's views (history only, all tests, 2 random subsets;
  seed 0, 7,000-token budget), exactly the rows the H2 HA was trained on;
- ``val.pt``: the H2 validation views (seed 10,000, 1 random subset);
- ``test/<arm>.pt``: the fixed test records of each arm (tests copied from an existing run);
- ``actions.pt``: the four diseases.

Each file holds ``x`` (float16, one 4,096-dim vector per record), ``y`` (disease index) and
bookkeeping. Finished files are skipped, so a resubmitted job continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.contrastive_ha.backbone import Backbone
from src.ha.prompts import ANSWER_PREFIX, DISEASES, chat_prompt, encode, ha_user_message

REPO = Path(__file__).resolve().parents[2]
H2_ADAPTER = REPO / "outputs/ha_sft/seed_0/adapter"


def action_prompt_ids(tokenizer, disease: str) -> list[int]:
    return encode(tokenizer, chat_prompt(tokenizer, f"Final diagnosis: {disease}"))


def state_ids_from_example(example: dict, answer_prefix: list[int]) -> list[int]:
    """The H2 training row's prompt + ``Hypothesis:`` (the scored prompt of src/score_ha.py)."""
    prompt_len = next(i for i, label in enumerate(example["labels"]) if label != -100)
    return example["input_ids"][:prompt_len] + answer_prefix


def embed_all(backbone: Backbone, prompts: list[list[int]], name: str) -> torch.Tensor:
    vectors, started = [], time.time()
    for i, ids in enumerate(prompts):
        vectors.append(backbone.embed(ids).half())
        if (i + 1) % 200 == 0 or i + 1 == len(prompts):
            print(f"[embed {name}] {i + 1}/{len(prompts)} ({time.time() - started:.0f}s)", flush=True)
    return torch.stack(vectors) if vectors else torch.zeros(0)


def main() -> None:
    from src.ha.sft_data import build_examples
    from src.score_ha import first_episodes
    from src.train_ha_sft import load_split

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, choices=("base", "h2"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=EPISODES_CSV")
    parser.add_argument("--limit", type=int, default=None, help="Patients per split/arm (smoke test)")
    parser.add_argument("--max-tokens", type=int, default=7000)
    args = parser.parse_args()

    dataset_cfg = OmegaConf.load(REPO / "configs/dataset/defaults.yaml")
    model_cfg = OmegaConf.load(REPO / "configs/model/defaults.yaml")
    env_cfg = OmegaConf.load(REPO / "configs/environment/la_cdm.yaml")
    template = (REPO / env_cfg.hypothesis_prompt_template_file).read_text()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    answer_prefix = encode(tokenizer, ANSWER_PREFIX)
    out = args.out_dir
    (out / "test").mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="sdpa")
    if args.backbone == "h2":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(H2_ADAPTER))
    backbone = Backbone(model.eval(), args.backbone)
    meta = {"backbone": args.backbone, "model": model_cfg.model_name,
            "h2_adapter": str(H2_ADAPTER) if args.backbone == "h2" else None, "pooling": "last token, final layer",
            "limit": args.limit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    if not (out / "actions.pt").exists():
        actions = torch.stack([backbone.embed(action_prompt_ids(tokenizer, d)) for d in DISEASES])
        torch.save({"x": actions, "diseases": list(DISEASES)}, out / "actions.pt")

    for split, seed, n_random in (("val", 10_000, 1), ("train", 0, 2)):
        path = out / f"{split}.pt"
        if path.exists():
            print(f"[embed] {path} exists; skipped", flush=True)
            continue
        patients = load_split(dataset_cfg, split, args.limit)
        examples = build_examples(tokenizer, template, patients, seed, n_random, args.max_tokens)
        prompts = [state_ids_from_example(e, answer_prefix) for e in examples]
        x = embed_all(backbone, prompts, split)
        torch.save({"x": x, "y": torch.tensor([DISEASES.index(e["label"]) for e in examples]),
                    "patient_id": [e["patient_id"] for e in examples],
                    "num_tests": [len(e["tests"]) for e in examples],
                    "prompt_tokens": [len(p) for p in prompts]}, path)

    patients = {str(p["patient_id"]): p for p in load_split(dataset_cfg, "test", None)}
    for spec in args.arm:
        name, _, csv_path = spec.partition("=")
        path = out / "test" / f"{name}.pt"
        if path.exists():
            print(f"[embed] {path} exists; skipped", flush=True)
            continue
        sources = first_episodes(Path(csv_path))[: args.limit]
        prompts, labels = [], []
        for source in sources:
            patient = patients[str(source["patient_id"])]
            message = ha_user_message(template, patient["patient_history"], patient["test_results"], source["tests"])
            prompts.append(encode(tokenizer, chat_prompt(tokenizer, message)) + answer_prefix)
            labels.append(DISEASES.index(str(patient["condition"]).lower()))
        x = embed_all(backbone, prompts, name)
        torch.save({"x": x, "y": torch.tensor(labels), "patient_id": [s["patient_id"] for s in sources],
                    "source": csv_path, "prompt_tokens": [len(p) for p in prompts]}, path)
    print(f"[embed] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
