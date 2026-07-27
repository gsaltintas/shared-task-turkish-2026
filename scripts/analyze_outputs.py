#!/usr/bin/env python3
"""
Written with Claude Code

Compare Gemini responses in output/*.tsv against the gold `Answer` column
and report match accuracy overall and by category, under a few different
normalization strategies (raw / lowercase+punct-stripped / diacritics-stripped).

Usage:
    python scripts/analyze_outputs.py
    python scripts/analyze_outputs.py output/output_36_flash.tsv output/output_eng_prompt_25_lite.tsv output/output_tr_prompt_25_lite.tsv
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

GOLD_COLUMN_CANDIDATES = ["Answer Corrected Translation", "Answer"]


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str, *, strip_accents: bool) -> str:
    # Türkçe-uyumlu küçük harf: 'İ'->'i', 'I'->'ı' (Python .lower() bunları yanlış çevirir)
    text = text.strip().replace("İ", "i").replace("I", "ı").lower()
    if strip_accents:
        text = strip_diacritics(text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_gold_value(row: dict) -> str:
    for name in GOLD_COLUMN_CANDIDATES:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def find_response_column(fieldnames: list[str]) -> str | None:
    candidates = [f for f in fieldnames if "response" in f.lower()]
    if candidates:
        return candidates[0]
    return fieldnames[-1] if fieldnames else None


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


VARIANTS = ["raw", "normalized", "normalized_no_diacritics", "soft (substring)"]


def match_variants(gold: str, pred: str) -> dict[str, bool]:
    raw_match = pred.strip() == gold.strip()

    norm_gold = normalize(gold, strip_accents=False)
    norm_pred = normalize(pred, strip_accents=False)
    norm_match = norm_pred == norm_gold

    ascii_gold = normalize(gold, strip_accents=True)
    ascii_pred = normalize(pred, strip_accents=True)
    ascii_match = ascii_pred == ascii_gold

    # gold'un TAMAMI yanıtta geçiyorsa doğru; kısmi eşleşme (pred, gold'un parçası) sayılmaz
    soft_match = bool(ascii_gold) and (ascii_gold in ascii_pred)

    return {
        "raw": raw_match,
        "normalized": norm_match,
        "normalized_no_diacritics": ascii_match,
        "soft (substring)": soft_match,
    }


def analyze_file(path: Path) -> None:
    rows = load_rows(path)
    if not rows:
        print(f"{path.name}: no rows")
        return

    fieldnames = list(rows[0].keys())
    response_col = find_response_column(fieldnames)

    if response_col is None or not any(c in fieldnames for c in GOLD_COLUMN_CANDIDATES):
        print(f"{path.name}: missing response/gold column (found {fieldnames})")
        return

    total = 0
    errors = 0
    totals = {v: 0 for v in VARIANTS}
    by_category = defaultdict(lambda: {"total": 0, **{v: 0 for v in VARIANTS}})

    for row in rows:
        gold = get_gold_value(row)
        pred = (row.get(response_col) or "").strip()
        if not gold:
            continue

        total += 1
        category = row.get("Category") or "Unknown"
        by_category[category]["total"] += 1

        if pred.startswith("ERROR:"):
            errors += 1
            continue

        results = match_variants(gold, pred)
        for variant, is_match in results.items():
            if is_match:
                totals[variant] += 1
                by_category[category][variant] += 1

    print(f"\n=== {path.name} ===")
    print(f"response column: '{response_col}'  gold column: one of {GOLD_COLUMN_CANDIDATES}")
    print(f"rows with gold answer: {total}  |  errors: {errors}")
    for variant in VARIANTS:
        if total:
            print(f"  {variant:26s} {totals[variant]:3d}/{total} ({totals[variant] / total:.1%})")

    print("by category (raw / normalized / no-diacritics / soft):")
    for category, stats in sorted(by_category.items(), key=lambda kv: -kv[1]["total"]):
        t = stats["total"]
        if t == 0:
            continue
        parts = "  ".join(f"{stats[v]:3d}/{t:<3d}" for v in VARIANTS)
        print(f"  {category:20s} {parts}")


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(Path("output").glob("*.tsv"))

    if not paths:
        print("No output TSV files found.")
        return 1

    for path in paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        analyze_file(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
