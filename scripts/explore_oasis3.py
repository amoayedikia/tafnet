#!/usr/bin/env python3
"""
explore_oasis3.py
=================
A standalone explorer for an OASIS-3 clinical data export.

Expected layout (matches the OASIS3_data_files distribution):

    OASIS3_data_files/
        MRI-json/            -> per-scan JSON sidecars
        pychometrics/        -> psychometric / cognitive battery CSV(s)
        UDSa1/csv/           -> UDS form A1  (subject demographics)
        UDSb4/csv/           -> UDS form B4  (CDR - global / sum of boxes)
        UDSb5/csv/           -> UDS form B5  (NPI-Q behavioural inventory)
        UDSb6/csv/           -> UDS form B6  (GDS depression scale)
        UDSb7/csv/           -> UDS form B7  (FAQ functional assessment)
        UDSd1/csv/           -> UDS form D1  (clinician diagnosis)

The script:
  1. Inventories every CSV and JSON file under the root (size + counts).
  2. For each CSV: reports shape, dtypes, per-column missingness, and a
     preview; auto-detects the subject-ID and session/days columns and
     counts unique OASIS subjects and sessions.
  3. Summarises low-cardinality categorical columns (useful for CDR / Dx codes).
  4. Parses a sample of the MRI JSON sidecars and reports their key structure.
  5. Builds a cross-form subject-overlap table so you can see which subjects
     appear in which form.
  6. Optionally writes a Markdown report to disk.

Usage
-----
    python explore_oasis3.py
    python explore_oasis3.py --root "${OASIS3_ROOT}/OASIS3_data_files"
    python explore_oasis3.py --preview-rows 10 --max-categorical 25
    python explore_oasis3.py --save-report oasis3_overview.md

Dependencies: pandas (everything else is stdlib).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _resolve(p):
    """Expand ${ENV_VAR} and ~ in a user-supplied path, returning a Path."""
    import os
    from pathlib import Path as _P
    return _P(os.path.expandvars(str(p))).expanduser()


# OASIS-3 subject identifiers look like OAS3 followed by four digits, e.g. OAS30001.
OASIS_ID_RE = re.compile(r"OAS3\d{4}", re.IGNORECASE)

# Column-name hints used when auto-detecting structural columns.
ID_NAME_HINTS = ("oasis", "subject", "adrc", "label", "id")
DAYS_NAME_HINTS = ("days", "day_to", "days_to", "ses", "visit", "age")


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TB"


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV, falling back through a few encodings OASIS files sometimes use."""
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception) as err:  # noqa: BLE001
            last_err = err
            continue
    raise RuntimeError(f"Could not read {path}: {last_err}")


def detect_id_column(df: pd.DataFrame) -> str | None:
    """
    Find the column most likely to hold OASIS subject IDs.

    Strategy: scan object/string columns, count cells matching OAS3xxxx,
    and return the column with the most matches. Falls back to name hints.
    """
    best_col, best_hits = None, 0
    for col in df.columns:
        series = df[col].astype("string")
        hits = series.str.contains(OASIS_ID_RE, na=False).sum()
        if hits > best_hits:
            best_col, best_hits = col, int(hits)
    if best_col is not None and best_hits > 0:
        return best_col

    # Fallback: first column whose name looks like an identifier.
    for col in df.columns:
        if any(hint in col.lower() for hint in ID_NAME_HINTS):
            return col
    return None


def detect_days_column(df: pd.DataFrame) -> str | None:
    """Find a 'days from baseline' / session-offset column if present."""
    for col in df.columns:
        if any(hint in col.lower() for hint in DAYS_NAME_HINTS):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def extract_subjects(df: pd.DataFrame, id_col: str | None) -> set[str]:
    """Pull the set of unique OAS3xxxx subject IDs from the detected ID column."""
    if id_col is None:
        return set()
    joined = df[id_col].astype("string").fillna("")
    found: set[str] = set()
    for value in joined:
        found.update(m.group(0).upper() for m in OASIS_ID_RE.finditer(value))
    return found


# --------------------------------------------------------------------------- #
# Reporting blocks (each returns a list of Markdown/plain lines)
# --------------------------------------------------------------------------- #
def report_inventory(root: Path) -> list[str]:
    lines = ["## File inventory", ""]
    csv_files = sorted(root.rglob("*.csv"))
    json_files = sorted(root.rglob("*.json"))

    lines.append(f"- CSV files found: {len(csv_files)}")
    lines.append(f"- JSON files found: {len(json_files)}")
    lines.append("")

    lines.append("### Top-level folders")
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        n_csv = len(list(child.rglob("*.csv")))
        n_json = len(list(child.rglob("*.json")))
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        lines.append(
            f"- `{child.name}/`  ->  {n_csv} csv, {n_json} json, {human_size(size)}"
        )
    lines.append("")
    return lines


def report_csv(path: Path, root: Path, preview_rows: int, max_categorical: int):
    """Return (markdown_lines, subjects_set, form_label)."""
    rel = path.relative_to(root)
    form_label = path.parent.parent.name if path.parent.name == "csv" else path.stem

    lines = [f"### `{rel}`  (form: {form_label})", ""]
    try:
        df = read_csv_robust(path)
    except Exception as err:  # noqa: BLE001
        lines.append(f"  ! Failed to read: {err}")
        lines.append("")
        return lines, set(), form_label

    n_rows, n_cols = df.shape
    id_col = detect_id_column(df)
    days_col = detect_days_column(df)
    subjects = extract_subjects(df, id_col)

    lines.append(f"- Shape: {n_rows:,} rows x {n_cols} columns")
    lines.append(f"- Detected subject-ID column: {id_col!r}")
    lines.append(f"- Detected session/days column: {days_col!r}")
    lines.append(f"- Unique OASIS subjects: {len(subjects):,}")
    if days_col is not None:
        lines.append(f"- Rows per subject (mean): {n_rows / max(len(subjects), 1):.2f}")
    lines.append("")

    # Column / dtype / missingness table.
    miss = df.isna().mean().mul(100).round(1)
    lines.append("- Columns (dtype, % missing):")
    for col in df.columns:
        lines.append(f"    - {col}: {df[col].dtype}, {miss[col]}% missing")
    lines.append("")

    # Low-cardinality categorical value counts (CDR scores, Dx codes, etc.).
    cat_cols = [
        c
        for c in df.columns
        if c != id_col and 1 < df[c].nunique(dropna=True) <= max_categorical
    ]
    if cat_cols:
        lines.append("- Categorical value distributions (low cardinality):")
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False).head(max_categorical)
            rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
            lines.append(f"    - {col}: {rendered}")
        lines.append("")

    # Preview.
    if preview_rows > 0:
        lines.append(f"- Preview (first {preview_rows} rows):")
        with pd.option_context("display.max_columns", 12, "display.width", 160):
            preview = df.head(preview_rows).to_string(max_cols=12)
        lines.extend("    " + ln for ln in preview.splitlines())
        lines.append("")

    return lines, subjects, form_label


def report_json(root: Path, sample: int = 3) -> list[str]:
    json_files = sorted(root.rglob("*.json"))
    lines = ["## MRI JSON sidecars", ""]
    if not json_files:
        lines.append("No JSON files found.")
        lines.append("")
        return lines

    lines.append(f"- Total JSON sidecars: {len(json_files):,}")
    lines.append(f"- Inspecting first {min(sample, len(json_files))} as samples:")
    lines.append("")

    for path in json_files[:sample]:
        rel = path.relative_to(root)
        lines.append(f"### `{rel}`")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as err:  # noqa: BLE001
            lines.append(f"  ! Failed to parse: {err}")
            lines.append("")
            continue

        if isinstance(data, dict):
            lines.append(f"- Top-level keys ({len(data)}):")
            for key in list(data.keys())[:40]:
                val = data[key]
                preview = val if not isinstance(val, (dict, list)) else type(val).__name__
                lines.append(f"    - {key}: {preview}")
        elif isinstance(data, list):
            lines.append(f"- List of {len(data)} items; first item type: "
                         f"{type(data[0]).__name__ if data else 'n/a'}")
        lines.append("")
    return lines


def report_subject_overlap(form_subjects: dict[str, set[str]]) -> list[str]:
    lines = ["## Cross-form subject overlap", ""]
    if not form_subjects:
        lines.append("No subject sets were detected.")
        lines.append("")
        return lines

    all_subjects: set[str] = set()
    for subs in form_subjects.values():
        all_subjects |= subs
    lines.append(f"- Distinct subjects across all forms: {len(all_subjects):,}")
    lines.append("")
    lines.append("- Subjects present per form:")
    for form in sorted(form_subjects):
        lines.append(f"    - {form}: {len(form_subjects[form]):,}")
    lines.append("")

    # Subjects appearing in every form (intersection).
    common = set.intersection(*form_subjects.values()) if form_subjects else set()
    lines.append(f"- Subjects present in EVERY form: {len(common):,}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    default_root = (
        "${OASIS3_ROOT}/OASIS3_data_files"
    )
    parser = argparse.ArgumentParser(description="Explore an OASIS-3 data export.")
    parser.add_argument("--root", default=default_root, help="Path to OASIS3_data_files")
    parser.add_argument("--preview-rows", type=int, default=5,
                        help="Rows to show per CSV preview (0 to disable)")
    parser.add_argument("--max-categorical", type=int, default=20,
                        help="Max unique values for a column to be summarised")
    parser.add_argument("--json-samples", type=int, default=3,
                        help="Number of JSON sidecars to inspect")
    parser.add_argument("--save-report", default=None,
                        help="Optional path to write the full report as Markdown")
    args = parser.parse_args()

    root = _resolve(args.root)
    if not root.exists():
        print(f"ERROR: root path does not exist:\n  {root}", file=sys.stderr)
        print("Pass the correct location with --root", file=sys.stderr)
        return 1

    out: list[str] = [f"# OASIS-3 exploration report", "", f"Root: `{root}`", ""]
    out += report_inventory(root)

    # Walk every CSV, collecting subject sets per form.
    out.append("## Per-CSV summaries")
    out.append("")
    form_subjects: dict[str, set[str]] = {}
    for csv_path in sorted(root.rglob("*.csv")):
        block, subjects, form_label = report_csv(
            csv_path, root, args.preview_rows, args.max_categorical
        )
        out += block
        if subjects:
            form_subjects.setdefault(form_label, set()).update(subjects)

    out += report_subject_overlap(form_subjects)
    out += report_json(root, sample=args.json_samples)

    text = "\n".join(out)
    print(text)

    if args.save_report:
        report_path = _resolve(args.save_report)
        report_path.write_text(text, encoding="utf-8")
        print(f"\n[saved report -> {report_path}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
