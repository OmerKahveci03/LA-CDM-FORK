"""Split MIMIC-IV-Ext-CDM pickle files into stratified train/val/test CSVs.

Loads the four per-condition pickle files from MIMIC-IV-Ext-CDM, assigns
diagnosis labels, merges into a single DataFrame, and produces an 80/10/10
stratified split with a fixed random seed for reproducibility.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CONDITIONS = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
RANDOM_STATE = 269


def split_dataset(data_dir: Path, output_dir: Path) -> None:
    """Load per-condition pickle files, merge, and create stratified splits.

    Args:
        data_dir: Directory containing ``<condition>_hadm_info_first_diag.pkl``
            files from the MIMIC-IV-Ext-CDM download.
        output_dir: Directory to write ``train.csv``, ``val.csv``, ``test.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and label each condition
    dfs = []
    for condition in CONDITIONS:
        pkl_path = data_dir / f"{condition}_hadm_info_first_diag.pkl"
        print(f"Loading {pkl_path}")
        with open(pkl_path, "rb") as f:
            hadm_info = pickle.load(f)
        for pat in hadm_info.values():
            pat["Label"] = condition
        df = pd.DataFrame.from_dict(hadm_info, orient="index")
        dfs.append(df)
        print(f"  {condition}: {len(df)} patients")

    # Concatenate and set Patient ID from the dict keys (hadm_ids)
    df_all = pd.concat(dfs)
    df_all.reset_index(inplace=True)
    df_all.rename(columns={"index": "Patient ID"}, inplace=True)
    print(f"Total patients: {len(df_all)}")

    # Stratified split: 80% train, 10% val, 10% test
    train_df, remaining_df = train_test_split(
        df_all, test_size=0.2, stratify=df_all["Label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        remaining_df, test_size=0.5, stratify=remaining_df["Label"], random_state=RANDOM_STATE
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.reset_index(drop=True)
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"{split_name}: {len(split_df)} patients -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split MIMIC-IV-Ext-CDM into stratified train/val/test CSVs."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to MIMIC-IV-Ext-CDM directory containing pickle files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Output directory for CSV splits (default: data/).",
    )
    args = parser.parse_args()
    split_dataset(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
