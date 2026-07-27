#!/usr/bin/env python3
"""
Written with Claude Code

Run TSV questions through Gemini and write the responses to a new file.

The input file is expected to contain a `Question` column. The script preserves
all original columns and adds a `Gemini Response` column with the model output.

Usage:
    python scripts/gemini_csv_runner.py data/data.tsv -o output/output_gemini_25_flash_lite.tsv  --model=gemini-2.5-flash-lite
    python scripts/gemini_csv_runner.py data/data.tsv -o output/output_gemini_36_flash.tsv  --model="gemini-3.6-flash"

    python scripts/gemini_csv_runner.py data/data.tsv -o output/output_prompt_en_gemini_25_flash_lite.tsv  --model=gemini-2.5-flash-lite --prompt-language=en
    python scripts/gemini_csv_runner.py data/data.tsv -o output/output_prompt_en_gemini_36_flash.tsv  --model="gemini-3.6-flash" --prompt-language=en

Environment:
    GEMINI_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Literal

try:
    from google import genai
except ImportError as exc:  # pragma: no cover - import error shown at runtime
    raise SystemExit(
        "Missing dependency: google-genai. Install it using the project dependencies in pyproject.toml."
    ) from exc


DEFAULT_MODEL = "gemini-2.5-flash-lite"
QUESTION_COLUMN = "Question"


def make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)



def call_gemini(client: genai.Client, question: str, model: str, retries: int = 3, prompt_language: Literal["tr", "en"] = "tr") -> str:
    last_error: Exception | None = None
    if prompt_language == "en":
        prompt = (
        "Answer the question below. Return only the final answer, with no explanation, "
        "no markdown, and no extra text.\n\n"
        f"Question: {question}"
    )
    elif prompt_language == "tr":
        prompt = (
        "Aşağıdaki soruyu yanıtlayın. Açıklama, markdown veya ekstra metin olmadan sadece nihai cevabı döndürün.\n\n"
        f"Soru: {question}"
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = getattr(response, "text", None)
            if text is not None:
                return text.strip()
            return str(response).strip()
        except Exception as exc:  # noqa: BLE001 - surface API errors after retries
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    assert last_error is not None
    raise last_error


def process_csv(
    input_path: Path,
    output_path: Path,
    model: str,
    delimiter: str,
    num_examples: int | None = None,
    prompt_language: Literal["tr", "en"] = "tr",
) -> None:
    client = make_client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SystemExit("Input TSV does not contain a header row.")
        if QUESTION_COLUMN not in reader.fieldnames:
            raise SystemExit(f"Input TSV must contain a '{QUESTION_COLUMN}' column.")

        fieldnames = list(reader.fieldnames)
        OUTPUT_COLUMN = model + " Response"
        if OUTPUT_COLUMN not in fieldnames:
            fieldnames.append(OUTPUT_COLUMN)

        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()

            for row_number, row in enumerate(reader, start=2):
                if num_examples is not None and row_number > num_examples + 1:
                    break
                question = (row.get(QUESTION_COLUMN) or "").strip()
                if not question:
                    row[OUTPUT_COLUMN] = ""
                    writer.writerow(row)
                    continue

                try:
                    row[OUTPUT_COLUMN] = call_gemini(client, question, model=model, prompt_language=prompt_language)
                except Exception as exc:  # noqa: BLE001 - keep batch processing going
                    row[OUTPUT_COLUMN] = f"ERROR: {exc}"
                    print(f"Row {row_number}: {exc}", file=sys.stderr)

                writer.writerow(row)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Question column from a TSV through Gemini and save the responses."
    )
    parser.add_argument("input_tsv", type=Path, help="Path to the input TSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output TSV file (defaults to <input>_gemini.tsv)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Field delimiter to use for the input and output file (default: tab)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Number of examples to process (for testing; default: all)",
        default=None,
    )
    parser.add_argument(
        "--prompt-language",
        choices=["en", "tr"],
        default="tr",
        help="Language of the prompt to Gemini (default: tr)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input_tsv
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = args.output or input_path.with_name(f"{input_path.stem}_gemini.tsv")
    num_examples = args.num_examples
    process_csv(
        input_path,
        output_path,
        model=args.model,
        delimiter=args.delimiter,
        num_examples=num_examples,
        prompt_language=args.prompt_language,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
