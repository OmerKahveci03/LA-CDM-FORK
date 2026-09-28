"""Step 2: Run the vendored official MIMIC-CDM cohort builder."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


CONDITIONS = (
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
)

CONDITION_CONFIG = {
    "appendicitis": {
        "query": "acute appendicitis",
        "sanitize": ["acute appendicitis", "appendicitis", "appendectomy"],
    },
    "cholecystitis": {
        "query": "acute cholecystitis",
        "sanitize": ["acute cholecystitis", "cholecystitis", "cholecystostomy"],
    },
    "diverticulitis": {
        "query": "diverticulitis",
        "sanitize": ["acute diverticulitis", "diverticulitis"],
    },
    "pancreatitis": {
        "query": "acute pancreatitis",
        "sanitize": ["acute pancreatitis", "pancreatitis", "pancreatectomy"],
    },
}

MULTI_DIAGNOSIS_IDS = {26769588, 24309551, 20525915, 23074436}

INPUT_DIR = Path("/blue/prismap-ai-core/omerkahveci/mimic_cdm_candidates")
OUTPUT_DIR = Path("/blue/prismap-ai-core/omerkahveci/mimic_cdm_final")
BUILDER_DIR = (
    Path(__file__).resolve().parent / "MIMIC-Clinical-Decision-Making-Dataset"
)


def main() -> None:
    input_dir = INPUT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    builder_dir = BUILDER_DIR.resolve()
    create_dataset = builder_dir / "CreateDataset.py"

    for required in (input_dir / "hosp", input_dir / "note", create_dataset):
        if not required.exists():
            raise FileNotFoundError(f"Required path does not exist: {required}")

    expected_outputs = [
        output_dir / f"{condition}_hadm_info_first_diag.pkl"
        for condition in CONDITIONS
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite final cohort files: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "builder_work"
    work_dir.mkdir(exist_ok=True)

    # Import the reusable implementation from the official submodule. We do not
    # import CreateDataset itself because that file executes immediately on
    # import and has hard-coded empty path variables.
    sys.path.insert(0, str(builder_dir))
    from dataset.dataset import extract_hadm_ids, extract_info, load_data
    from dataset.labs import generate_lab_test_mapping
    from utils.nlp import extract_primary_diagnosis

    (
        admissions_df,
        transfers_df,
        diag_icd,
        procedures_df,
        discharge_df,
        radiology_report_df,
        radiology_report_details_df,
        lab_events_df,
        microbiology_df,
    ) = load_data(str(input_dir))

    # The imported extraction functions write human-readable diagnostic files
    # relative to cwd. Keep those artifacts out of the submodule.
    previous_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        for condition, config in CONDITION_CONFIG.items():
            hadm_ids = extract_hadm_ids(
                config["query"], diag_icd, discharge_df
            )
            _, clean_records = extract_info(
                hadm_ids,
                condition,
                config["sanitize"],
                discharge_df,
                admissions_df,
                transfers_df,
                lab_events_df,
                microbiology_df,
                radiology_report_df,
                radiology_report_details_df,
                diag_icd,
                procedures_df,
            )
            if clean_records is None:
                raise RuntimeError(f"Official extraction failed for {condition}")

            final_records = {}
            for hadm_id, record in clean_records.items():
                if hadm_id in MULTI_DIAGNOSIS_IDS:
                    continue
                primary = extract_primary_diagnosis(
                    record["Discharge Diagnosis"].lower()
                )
                if primary and condition in primary.lower():
                    final_records[hadm_id] = record

            destination = output_dir / f"{condition}_hadm_info_first_diag.pkl"
            with destination.open("wb") as file:
                pickle.dump(final_records, file)
            print(f"Wrote {len(final_records):,} {condition} cases: {destination}")

        generate_lab_test_mapping(str(input_dir / "hosp"))
        mapping_pickle = input_dir / "hosp" / "lab_test_mapping.pkl"
        with mapping_pickle.open("rb") as file:
            lab_mapping = pickle.load(file)
        lab_mapping.to_csv(output_dir / "lab_test_mapping.csv", index=False)
    finally:
        os.chdir(previous_cwd)

    missing = [path for path in expected_outputs if not path.exists()]
    if missing:
        raise RuntimeError(
            "Builder finished without expected output(s): "
            + ", ".join(str(path) for path in missing)
        )

    print("Final cohort created:")
    for path in expected_outputs:
        print(f"  {path}")
    print(f"  {output_dir / 'lab_test_mapping.csv'}")


if __name__ == "__main__":
    main()
