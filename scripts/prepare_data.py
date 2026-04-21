"""End-to-end data preparation for LA-CDM.

Takes the raw MIMIC-IV-Ext-CDM files as downloaded from PhysioNet and produces
the ``data/`` directory expected by the training pipeline::

    data/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── lab_test_mapping.csv

Pipeline steps:
    1. Split the four per-condition pickle files into stratified
       train / val / test CSVs (80 / 10 / 10, seed 269).
    2. Copy the lab test mapping CSV.
    3. Summarize each patient's history using an LLM, adding a
       ``Patient History Summary`` column.

Requires a GPU for step 3.  Run via SLURM::

    srun --gres=gpu:1 python scripts/prepare_data.py \\
        --data_dir /path/to/mimic-iv-ext-cdm/1.1 \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import pickle
import shutil
from pathlib import Path

from split_dataset import split_dataset
from create_data_with_summaries import load_summarizer, summarize_histories


def copy_lab_test_mapping(data_dir: Path, output_dir: Path) -> None:
    """Copy (or convert) the lab test mapping into the output directory.

    Prefers the CSV version if available; falls back to converting the
    pickle file.

    Args:
        data_dir: MIMIC-IV-Ext-CDM download directory.
        output_dir: Target ``data/`` directory.
    """
    dst = output_dir / "lab_test_mapping.csv"
    csv_src = data_dir / "lab_test_mapping.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, dst)
        print(f"Copied {csv_src} -> {dst}")
        return

    pkl_src = data_dir / "lab_test_mapping.pkl"
    print(f"CSV not found, converting {pkl_src}")
    with open(pkl_src, "rb") as f:
        lab_test_mapping = pickle.load(f)
    lab_test_mapping.to_csv(dst, index=False)
    print(f"Saved {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LA-CDM data from a raw MIMIC-IV-Ext-CDM download."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to MIMIC-IV-Ext-CDM directory (contains pickle files).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/",
        help="Output directory (default: data/).",
    )
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="HuggingFace model for patient history summarization.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens per summary (default: 512).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Split dataset
    print("=" * 60)
    print("Step 1: Splitting dataset into train/val/test")
    print("=" * 60)
    split_dataset(data_dir, output_dir)

    # Step 2: Copy lab test mapping
    print("\n" + "=" * 60)
    print("Step 2: Copying lab test mapping")
    print("=" * 60)
    copy_lab_test_mapping(data_dir, output_dir)

    # Step 3: Summarize patient histories
    print("\n" + "=" * 60)
    print("Step 3: Summarizing patient histories")
    print("=" * 60)
    model, tokenizer = load_summarizer(args.model)
    for split in ["train", "val", "test"]:
        csv_path = output_dir / f"{split}.csv"
        print(f"\nProcessing {split} split...")
        summarize_histories(
            csv_path, csv_path, args.model, args.max_new_tokens,
            model=model, tokenizer=tokenizer,
        )

    print("\n" + "=" * 60)
    print(f"Done. Data files saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
