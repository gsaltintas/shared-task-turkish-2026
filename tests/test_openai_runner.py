#!/usr/bin/env python3
"""
openai_csv_runner.py için ağ/GPU/anahtar GEREKTİRMEYEN doğrulama testleri.

Çalıştırma:
    /opt/Anaconda-2021.05/bin/python3 tests/test_openai_runner.py

T1 · Mock OpenAI-uyumlu sunucuyla uçtan uca runner testi.
T2 · Minimal-diff kanıtı (prompt_tr / döngü / OUTPUT_COLUMN birebir; Gemini SDK yok).
T3 · analyze_outputs.py uyumu (+ match_variants/normalize birim assert'leri).
"""
from __future__ import annotations

import csv
import http.server
import io
import json
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import openai_csv_runner as R  # noqa: E402
import analyze_outputs as A    # noqa: E402

# Yavaş retry beklemelerini testte devre dışı bırak.
R.time.sleep = lambda *_a, **_k: None  # type: ignore

_failures = []

def check(name, got, exp):
    ok = got == exp
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: got={got!r} exp={exp!r}")
    if not ok:
        _failures.append(name)

def check_true(name, cond):
    print(f"[{'OK ' if cond else 'FAIL'}] {name}")
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# Mock OpenAI-uyumlu sunucu
# --------------------------------------------------------------------------- #
ANSWERS = {
    "Fatih Terim'in lakabı?": "İmparator",       # doğru olacak
    "Naim Süleymanoğlu'nun lakabı?": "Yanlış",   # yanlış olacak
}

class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # sessiz
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        content = req["messages"][0]["content"]
        question = content.split("Soru:", 1)[-1].strip()
        if "TRIGGER_ERROR" in question:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"boom"}')
            return
        answer = ANSWERS.get(question, "BILINMIYOR")
        data = json.dumps({"choices": [{"message": {"content": answer}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_mock():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --------------------------------------------------------------------------- #
# T1 · uçtan uca
# --------------------------------------------------------------------------- #
def test_t1_end_to_end(tmp):
    print("\n--- T1: mock sunucu uçtan uca ---")
    server, port = start_mock()
    try:
        inp = tmp / "mini.tsv"
        out = tmp / "mini_out.tsv"
        rows = [
            {"ID": "1", "Question": "Fatih Terim'in lakabı?", "Answer": "İmparator"},
            {"ID": "2", "Question": "Naim Süleymanoğlu'nun lakabı?", "Answer": "Cep Herkülü"},
            {"ID": "3", "Question": "TRIGGER_ERROR", "Answer": "X"},
            {"ID": "4", "Question": "", "Answer": "Y"},  # boş soru -> atlanır
        ]
        write_tsv(inp, rows, ["ID", "Question", "Answer"])

        rc = R.main([
            str(inp), "-o", str(out),
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--api-key-env", "NONE",
            "--model", "test-model",
        ])
        check("T1 exit code", rc, 0)

        got = read_tsv(out)
        resp_col = "test-model Response"
        check_true("T1 Response kolonu var", resp_col in got[0])
        check("T1 row1 (doğru cevap)", got[0][resp_col], "İmparator")
        check("T1 row2 (yanlış cevap)", got[1][resp_col], "Yanlış")
        check_true("T1 row3 ERROR: ile başlar", got[2][resp_col].startswith("ERROR:"))
        check("T1 row4 boş soru -> boş", got[3][resp_col], "")
        # orijinal kolonlar korunuyor
        check_true("T1 orijinal kolonlar korundu", all(k in got[0] for k in ("ID", "Question", "Answer")))

        # --num-examples 1 -> sadece 1 veri satırı yazılır
        out2 = tmp / "mini_n1.tsv"
        R.main([
            str(inp), "-o", str(out2),
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--api-key-env", "NONE", "--model", "test-model",
            "--num-examples", "1",
        ])
        check("T1 --num-examples 1 satır sayısı", len(read_tsv(out2)), 1)
        return out
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- #
# T2 · minimal-diff kanıtı
# --------------------------------------------------------------------------- #
def test_t2_minimal_diff():
    print("\n--- T2: minimal-diff kanıtı ---")
    gem = (SCRIPTS / "gemini_csv_runner.py").read_text(encoding="utf-8")
    new = (SCRIPTS / "openai_csv_runner.py").read_text(encoding="utf-8")

    prompt_block = (
        '    prompt_tr = (\n'
        '        "Aşağıdaki soruyu yanıtlayın. Açıklama, markdown veya ekstra metin olmadan sadece nihai cevabı döndürün.\\n\\n"\n'
        '        f"Soru: {question}"\n'
        '    )\n'
        '    prompt = prompt_tr\n'
    )
    check_true("T2 prompt_tr bloğu Gemini'de var", prompt_block in gem)
    check_true("T2 prompt_tr bloğu kopyada BİREBİR", prompt_block in new)

    anchors = [
        'QUESTION_COLUMN = "Question"',
        '        OUTPUT_COLUMN = model + " Response"',
        '            for row_number, row in enumerate(reader, start=2):',
        '                if num_examples is not None and row_number > num_examples + 1:',
        '                question = (row.get(QUESTION_COLUMN) or "").strip()',
        '                if not question:',
        '                    row[OUTPUT_COLUMN] = ""',
    ]
    for a in anchors:
        check_true(f"T2 anchor birebir: {a.strip()[:42]}", (a in gem) and (a in new))

    # Kopyada Gemini SDK yok; OpenAI yolu var
    check_true("T2 kopyada 'genai' YOK", "genai" not in new and "google" not in new)
    check_true("T2 kopyada call_model var", "def call_model(" in new)
    check_true("T2 kopyada chat/completions var", "/chat/completions" in new)
    check_true("T2 kopyada --base-url var", "--base-url" in new)
    check_true("T2 kopyada --api-key-env var", "--api-key-env" in new)
    # sampling parametresi göndermiyoruz: payload'da JSON anahtarı olarak YOK
    # (docstring'de kelime olarak geçebilir; önemli olan gövdede key olmaması)
    check_true(
        "T2 payload'da sampling paramı YOK",
        '"temperature"' not in new and '"max_tokens"' not in new and '"top_p"' not in new,
    )


# --------------------------------------------------------------------------- #
# T3 · analyze_outputs uyumu + birim
# --------------------------------------------------------------------------- #
def test_t3_analyze(t1_out):
    print("\n--- T3: analyze_outputs uyumu + birim ---")
    # birim: match_variants
    check("T3 raw eşit", A.match_variants("Baklava", "Baklava")["raw"], True)
    check("T3 raw farklı-case False", A.match_variants("baklava", "Baklava")["raw"], False)
    check("T3 normalized case", A.match_variants("baklava", "Baklava")["normalized"], True)
    check("T3 normalized noktalama", A.match_variants("Zeybek!", "zeybek")["normalized"], True)
    check("T3 soft substring", A.match_variants("İstanbul", "İstanbul'dur")["soft (substring)"], True)

    # uyum: analyze_file T1 çıktısını sorunsuz işleyebilmeli, kolonu bulmalı
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.analyze_file(Path(t1_out))
    text = buf.getvalue()
    check_true("T3 analyze response kolonunu buldu", "response column: 'test-model Response'" in text)
    check_true("T3 analyze accuracy bastı", "rows with gold answer:" in text)
    print(text.strip())


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        t1_out = test_t1_end_to_end(tmp)
        test_t2_minimal_diff()
        test_t3_analyze(t1_out)
    print("\n==============================")
    if _failures:
        print(f"BAŞARISIZ ❌  ({len(_failures)} test): {_failures}")
        return 1
    print("TÜM TESTLER GEÇTİ ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
