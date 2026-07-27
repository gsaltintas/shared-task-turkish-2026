#!/usr/bin/env python3
"""
Written with Claude Code

Run TSV questions through an OpenAI-compatible chat endpoint (our vLLM server /
DeepSeek / any /v1/chat/completions server) and write the responses to a new file.

`gemini_csv_runner.py`'nin minimal-diff kopyasıdır: TSV okuma/yazma döngüsü, Türkçe
prompt (`prompt_tr`) ve çıktı kolonu (`<model> Response`) BİREBİR aynıdır; yalnızca
model çağrısı Google GenAI SDK yerine OpenAI-uyumlu HTTP çağrısına çevrildi. Hiçbir
sampling parametresi gönderilmez (temperature/max_tokens yok → model defaultları).

The input file is expected to contain a `Question` column. The script preserves
all original columns and adds a `<model> Response` column with the model output.

Usage:
    # Qwen (kendi vLLM sunucumuz; açık server, anahtar yok)
    python scripts/openai_csv_runner.py data/data.tsv -o output/output_qwen36.tsv \
        --base-url http://<host>:8765/v1 --api-key-env NONE --model Qwen/Qwen3.6-35B-A3B

    # DeepSeek V4 Flash (API)
    export DEEPSEEK_API_KEY=...
    python scripts/openai_csv_runner.py data/data.tsv -o output/output_deepseek.tsv \
        --base-url https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY --model deepseek-v4-flash

Environment:
    --api-key-env ile verilen ortam değişkeni (ör. DEEPSEEK_API_KEY) set edilmeli;
    açık (anahtarsız) sunucu için --api-key-env NONE.
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
    import requests  # type: ignore

    def _http_post_json(url, headers, payload, timeout=120.0):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

except ImportError:  # pragma: no cover - requests yoksa urllib'e düş
    import json as _json
    import urllib.request

    def _http_post_json(url, headers, payload, timeout=120.0):
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode("utf-8"))


QUESTION_COLUMN = "Question"


def call_model(
    question: str, model: str, base_url: str, api_key: str | None, retries: int = 3, prompt_language: Literal["tr", "en"] = "tr"
) -> str:
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

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(1, retries + 1):
        try:
            data = _http_post_json(url, headers, payload)
            message = data["choices"][0].get("message") or {}
            text = message.get("content") or message.get("reasoning_content")
            if text is not None:
                return text.strip()
            return str(data).strip()
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
    base_url: str,
    api_key: str | None,
    delimiter: str,
    num_examples: int | None = None,
    prompt_language: Literal["tr", "en"] = "tr",
) -> None:
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
                    row[OUTPUT_COLUMN] = call_model(
                        question, model=model, base_url=base_url, api_key=api_key, prompt_language=prompt_language
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch processing going
                    row[OUTPUT_COLUMN] = f"ERROR: {exc}"
                    print(f"Row {row_number}: {exc}", file=sys.stderr)

                writer.writerow(row)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Question column from a TSV through an OpenAI-compatible endpoint and save the responses."
    )
    parser.add_argument("input_tsv", type=Path, help="Path to the input TSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output TSV file (defaults to <input>_openai.tsv)",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://host:8765/v1 or https://api.deepseek.com/v1",
    )
    parser.add_argument(
        "--api-key-env",
        default="NONE",
        help="Env var name holding the API key (e.g. DEEPSEEK_API_KEY). Use NONE for an open server.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name/id to send to the endpoint",
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
        help="Language of the prompt to the model (default: tr)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input_tsv
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    env = args.api_key_env
    if env and env.upper() != "NONE":
        api_key = os.environ.get(env)
        if not api_key:
            raise SystemExit(f"{env} is not set in the environment (or pass --api-key-env NONE).")
    else:
        api_key = None

    output_path = args.output or input_path.with_name(f"{input_path.stem}_openai.tsv")
    num_examples = args.num_examples
    process_csv(
        input_path,
        output_path,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        delimiter=args.delimiter,
        num_examples=num_examples,
        prompt_language=args.prompt_language,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
