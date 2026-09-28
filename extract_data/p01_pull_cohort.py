"""Step 1: Extract the raw MIMIC-IV rows needed to build the MIMIC-CDM cohort.

This is a *candidate cohort* extractor.  It identifies hospital admissions whose
ICD diagnosis title contains one of the four MIMIC-CDM diseases, requires a
matching discharge note, and then streams the relevant MIMIC-IV and
MIMIC-IV-Note tables into a much smaller directory.

The output intentionally mirrors the layout expected by the official
MIMIC-Clinical-Decision-Making-Dataset ``CreateDataset.py`` script::

    output/
    |-- hosp/
    |   |-- admissions.csv
    |   |-- diagnoses_icd.csv
    |   `-- ...
    `-- note/
        |-- discharge.csv
        |-- radiology.csv
        `-- radiology_detail.csv

The official builder must still be run afterward.  It extracts HPI/physical
exam sections, checks data completeness and abdominal imaging, removes label
leakage, and writes the four ``*_hadm_info_first_diag.pkl`` files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CONDITIONS = {
    "appendicitis": "acute appendicitis",
    "cholecystitis": "acute cholecystitis",
    "diverticulitis": "diverticulitis",
    "pancreatitis": "acute pancreatitis",
}

HOSP_EVENT_TABLES = (
    "admissions",
    "transfers",
    "diagnoses_icd",
    "procedures_icd",
    "labevents",
    "microbiologyevents",
)

HOSP_LOOKUP_TABLES = (
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_labitems",
)

NOTE_TABLES = (
    "discharge",
    "radiology",
    "radiology_detail",
)


def input_path(directory: Path, stem: str) -> Path:
    """Return either ``<stem>.csv.gz`` or ``<stem>.csv`` from a directory."""
    for suffix in (".csv.gz", ".csv"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {stem}.csv.gz (or .csv) under {directory}")


def read_table(directory: Path, stem: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(input_path(directory, stem), **kwargs)


def write_chunks(chunks: Iterable[pd.DataFrame], destination: Path) -> int:
    """Write chunks to one CSV and return the number of output rows."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    row_count = 0

    for chunk in chunks:
        if chunk.empty:
            continue
        chunk.to_csv(
            destination,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        row_count += len(chunk)

    if not wrote_header:
        # Preserve the input schema even when no rows match.
        raise RuntimeError(f"No matching rows were found for {destination.name}")
    return row_count


def find_candidate_admissions(hosp_dir: Path) -> tuple[set[int], dict[str, int]]:
    diagnoses = read_table(
        hosp_dir,
        "diagnoses_icd",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": "string"},
    ).dropna(subset=["hadm_id", "icd_code", "icd_version"])
    descriptions = read_table(
        hosp_dir,
        "d_icd_diagnoses",
        usecols=["icd_code", "icd_version", "long_title"],
        dtype={"icd_code": "string", "long_title": "string"},
    ).dropna(subset=["icd_code", "icd_version", "long_title"])

    merged = diagnoses.merge(
        descriptions,
        on=["icd_code", "icd_version"],
        how="left",
        validate="many_to_one",
    )

    all_ids: set[int] = set()
    counts: dict[str, int] = {}
    for condition, phrase in CONDITIONS.items():
        mask = merged["long_title"].str.contains(phrase, case=False, na=False)
        ids = set(merged.loc[mask, "hadm_id"].astype("int64"))
        counts[condition] = len(ids)
        all_ids.update(ids)
    return all_ids, counts


def discharge_linked_admissions(note_dir: Path, candidates: set[int]) -> set[int]:
    linked: set[int] = set()
    path = input_path(note_dir, "discharge")
    for chunk in pd.read_csv(path, usecols=["hadm_id"], chunksize=250_000):
        values = chunk["hadm_id"].dropna().astype("int64")
        linked.update(values[values.isin(candidates)].tolist())
    return linked


def candidate_subjects(hosp_dir: Path, hadm_ids: set[int]) -> set[int]:
    admissions = read_table(
        hosp_dir, "admissions", usecols=["subject_id", "hadm_id"]
    ).dropna(subset=["subject_id", "hadm_id"])
    rows = admissions[admissions["hadm_id"].astype("int64").isin(hadm_ids)]
    return set(rows["subject_id"].astype("int64"))


def filtered_chunks(
    source: Path,
    hadm_ids: set[int],
    subject_ids: set[int],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Yield rows connected to candidate admissions.

    Rows with a candidate ``hadm_id`` are retained. Rows lacking ``hadm_id``
    are retained when their ``subject_id`` belongs to a candidate patient; the
    official builder later associates these events using admission times.
    """
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        mask = pd.Series(False, index=chunk.index)
        if "hadm_id" in chunk.columns:
            numeric_hadm = pd.to_numeric(chunk["hadm_id"], errors="coerce")
            mask |= numeric_hadm.isin(hadm_ids)
            if "subject_id" in chunk.columns:
                numeric_subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
                mask |= numeric_hadm.isna() & numeric_subject.isin(subject_ids)
        elif "subject_id" in chunk.columns:
            numeric_subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
            mask |= numeric_subject.isin(subject_ids)
        else:
            raise ValueError(f"Cannot filter {source}: no hadm_id or subject_id column")
        yield chunk.loc[mask]


def copy_lookup_table(source_dir: Path, stem: str, output_dir: Path) -> int:
    source = input_path(source_dir, stem)
    destination = output_dir / f"{stem}.csv"
    frame = pd.read_csv(source, low_memory=False)
    frame.to_csv(destination, index=False)
    return len(frame)


def radiology_detail_ids(radiology_csv: Path, detail_source: Path) -> set[str]:
    """Collect selected radiology note IDs plus linked parent note IDs."""
    selected = set(
        pd.read_csv(radiology_csv, usecols=["note_id"])["note_id"]
        .dropna()
        .astype("string")
    )
    parent_ids: set[str] = set()
    for chunk in pd.read_csv(
        detail_source,
        usecols=["note_id", "field_name", "field_value"],
        chunksize=250_000,
        low_memory=False,
    ):
        rows = chunk[
            chunk["note_id"].astype("string").isin(selected)
            & chunk["field_name"].eq("parent_note_id")
        ]
        parent_ids.update(rows["field_value"].dropna().astype("string"))
    return selected | parent_ids


def note_id_chunks(
    source: Path, note_ids: set[str], chunksize: int
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        if "note_id" not in chunk.columns:
            raise ValueError(f"Cannot filter {source}: no note_id column")
        yield chunk[chunk["note_id"].astype("string").isin(note_ids)]


def extract(args: argparse.Namespace) -> None:
    mimic_dir = Path(args.mimic_dir).resolve()
    hosp_dir = mimic_dir / "hosp"
    note_dir = Path(args.note_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Unrelated files (including a copy of this script) may live beside the
    # extracted data. Refuse only when outputs managed by this script already
    # exist, so a partial or completed extraction is never silently appended to.
    managed_outputs = (
        output_dir / "hosp",
        output_dir / "note",
        output_dir / "extraction_metadata.json",
    )
    existing_outputs = [path for path in managed_outputs if path.exists()]
    if existing_outputs:
        formatted = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Refusing to overwrite existing extraction output(s): " + formatted
        )
    output_hosp = output_dir / "hosp"
    output_note = output_dir / "note"
    output_hosp.mkdir(parents=True, exist_ok=True)
    output_note.mkdir(parents=True, exist_ok=True)

    # Fail early before doing expensive work.
    for table in HOSP_EVENT_TABLES + HOSP_LOOKUP_TABLES:
        input_path(hosp_dir, table)
    for table in NOTE_TABLES:
        input_path(note_dir, table)

    candidate_hadm_ids, diagnosis_counts = find_candidate_admissions(hosp_dir)
    print(f"ICD-title candidates: {len(candidate_hadm_ids):,}")
    for condition, count in diagnosis_counts.items():
        print(f"  {condition}: {count:,}")

    hadm_ids = discharge_linked_admissions(note_dir, candidate_hadm_ids)
    subject_ids = candidate_subjects(hosp_dir, hadm_ids)
    print(f"Candidates with discharge notes: {len(hadm_ids):,}")
    print(f"Associated subjects: {len(subject_ids):,}")
    if not hadm_ids:
        raise RuntimeError("No candidate admissions have matching discharge notes")

    row_counts: dict[str, int] = {}
    for table in HOSP_EVENT_TABLES:
        source = input_path(hosp_dir, table)
        destination = output_hosp / f"{table}.csv"
        count = write_chunks(
            filtered_chunks(source, hadm_ids, subject_ids, args.chunksize),
            destination,
        )
        row_counts[f"hosp/{table}.csv"] = count
        print(f"Wrote {count:,} rows: {destination}")

    for table in HOSP_LOOKUP_TABLES:
        count = copy_lookup_table(hosp_dir, table, output_hosp)
        row_counts[f"hosp/{table}.csv"] = count
        print(f"Wrote {count:,} rows: {output_hosp / (table + '.csv')}")

    for table in ("discharge", "radiology"):
        source = input_path(note_dir, table)
        destination = output_note / f"{table}.csv"
        count = write_chunks(
            filtered_chunks(source, hadm_ids, subject_ids, args.chunksize),
            destination,
        )
        row_counts[f"note/{table}.csv"] = count
        print(f"Wrote {count:,} rows: {destination}")

    detail_source = input_path(note_dir, "radiology_detail")
    selected_note_ids = radiology_detail_ids(
        output_note / "radiology.csv", detail_source
    )
    detail_destination = output_note / "radiology_detail.csv"
    count = write_chunks(
        note_id_chunks(detail_source, selected_note_ids, args.chunksize),
        detail_destination,
    )
    row_counts["note/radiology_detail.csv"] = count
    print(f"Wrote {count:,} rows: {detail_destination}")

    metadata = {
        "mimic_dir": str(mimic_dir),
        "note_dir": str(note_dir),
        "conditions": CONDITIONS,
        "diagnosis_candidate_counts": diagnosis_counts,
        "unique_candidate_hadm_ids": len(hadm_ids),
        "unique_candidate_subject_ids": len(subject_ids),
        "row_counts": row_counts,
        "next_step": (
            "Run the official MIMIC-Clinical-Decision-Making-Dataset "
            "CreateDataset.py with base_mimic set to this output directory."
        ),
    }
    metadata_path = output_dir / "extraction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote extraction metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract candidate MIMIC-CDM admissions from MIMIC-IV 2.2."
    )
    parser.add_argument(
        "--mimic-dir",
        default="/orange/prismap-data-core/MIMIC/physionet.org/files/mimiciv/2.2",
        help="MIMIC-IV directory containing hosp/.",
    )
    parser.add_argument(
        "--note-dir",
        default=(
            "/orange/prismap-data-core/MIMIC/physionet.org/files/"
            "mimic-iv-note/2.2/note"
        ),
        help="MIMIC-IV-Note directory containing discharge and radiology files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/blue/prismap-ai-core/omerkahveci/mimic_cdm_candidates",
        help="New or empty output directory.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows read per chunk for large tables (default: 250000).",
    )
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    return args


if __name__ == "__main__":
    extract(parse_args())
