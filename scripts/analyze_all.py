#!/usr/bin/env python3
"""
analyze_all.py — TEK analiz kodu. Tüm çıktılar `analysis/` klasörüne AYRI TSV'ler olarak:

  analysis/model_correct.tsv             data.tsv kopyası + her model için '<tam ad> correct' (True/False)
  analysis/model_accuracy.tsv            Model | Correct | Total | Accuracy(%)
  analysis/category_accuracy.tsv         Category | Questions | Accuracy(%)   (tüm modeller ort.)
  analysis/model_x_category_accuracy.tsv Model × Kategori accuracy(%) matrisi

Doğruluk = REPONUN KENDİ analyze_outputs.py mantığı:
    match_variants(gold, response)["soft (substring)"]  (Türkçe `Answer` yanıtın içinde tam geçiyor mu?)
Eşleştirme `ID` ile. Tüm modeller (TR + EN promptlar dahil).

Kullanım: python scripts/analyze_all.py
"""
from __future__ import annotations

import csv
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import analyze_outputs as A  # repo'nun kendi kodu

REPO = SCRIPTS.parent
DATA = REPO / "data" / "data.tsv"
OUTDIR = REPO / "analysis"

KEY, GOLD, CATCOL = "ID", "Answer", "Category"
VARIANT = "soft (substring)"

# Tam model adı -> output tsv  (TR ve EN promptlar dahil TÜM modeller)
MODELS = OrderedDict([
    ("Qwen3.6-35B-A3B",             "output/output_qwen36.tsv"),
    ("deepseek-v4-flash",           "output/output_deepseek.tsv"),
    ("gemini-3.6-flash-tr",         "output/output_36_flash.tsv"),
    ("gemini-3.6-flash-en",         "output/output_prompt_en_gemini_36_flash.tsv"),
    ("gemini-2.5-flash-lite-tr",    "output/output_gemini_25_flash_lite.tsv"),
    ("gemini-2.5-flash-lite-en",    "output/output_prompt_en_gemini_25_flash_lite.tsv"),
])
COLNAME = {name: f"{name} correct" for name in MODELS}


def load_model(path: str):
    p = REPO / path
    if not p.exists():
        return None
    with p.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return {}
    rc = A.find_response_column(list(rows[0].keys()))
    return {(r.get(KEY) or "").strip(): (r.get(rc) or "") for r in rows}


def is_correct(gold: str, resp: str):
    if not (gold or "").strip():
        return None
    return A.match_variants(gold, resp)[VARIANT]


def acc(c, t):
    return round(100.0 * c / t, 1) if t else ""


def write_tsv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        base = list(csv.DictReader(f, delimiter="\t"))
    if not base:
        raise SystemExit(f"{DATA} boş.")
    fields = list(base[0].keys())

    resp = {name: load_model(path) for name, path in MODELS.items()}
    for name in MODELS:
        if resp[name] is None:
            print(f"UYARI: {MODELS[name]} yok → '{COLNAME[name]}' boş.", file=sys.stderr)
        if COLNAME[name] not in fields:
            fields.append(COLNAME[name])

    # correct kolonlarını doldur
    for row in base:
        rid = (row.get(KEY) or "").strip()
        gold = row.get(GOLD) or ""
        for name in MODELS:
            d = resp[name]
            if not d or rid not in d:
                row[COLNAME[name]] = ""
                continue
            c = is_correct(gold, d[rid])
            row[COLNAME[name]] = "" if c is None else str(c)

    # accuracy hesapları
    model_c = defaultdict(int); model_t = defaultdict(int)
    cat_count = defaultdict(int)
    cell_c = defaultdict(lambda: defaultdict(int))
    cell_t = defaultdict(lambda: defaultdict(int))
    for row in base:
        cat = (row.get(CATCOL) or "?").strip() or "?"
        cat_count[cat] += 1
        for name in MODELS:
            v = row.get(COLNAME[name])
            if v not in ("True", "False"):
                continue
            ok = 1 if v == "True" else 0
            model_c[name] += ok; model_t[name] += 1
            cell_c[name][cat] += ok; cell_t[name][cat] += 1

    cats = sorted(cat_count, key=lambda c: -cat_count[c])
    models_sorted = sorted(MODELS, key=lambda n: -(model_c[n] / model_t[n] if model_t[n] else 0))

    OUTDIR.mkdir(exist_ok=True)

    # 1) input kopyası + correct kolonları
    with (OUTDIR / "model_correct.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(base)

    # 2) model accuracy
    write_tsv(OUTDIR / "model_accuracy.tsv", ["Model", "Correct", "Total", "Accuracy(%)"],
              [[n, model_c[n], model_t[n], acc(model_c[n], model_t[n])] for n in models_sorted])

    # 3) kategori accuracy (tüm modeller ort.)
    crows = []
    for cat in cats:
        c = sum(cell_c[n][cat] for n in MODELS); t = sum(cell_t[n][cat] for n in MODELS)
        crows.append([cat, cat_count[cat], acc(c, t)])
    write_tsv(OUTDIR / "category_accuracy.tsv", ["Category", "Questions", "Accuracy(%)"], crows)

    # 4) model × kategori accuracy(%)
    mrows = [[n] + [acc(cell_c[n][cat], cell_t[n][cat]) for cat in cats] for n in models_sorted]
    write_tsv(OUTDIR / "model_x_category_accuracy.tsv", ["Model"] + cats, mrows)

    print(f"Yazıldı → {OUTDIR}/ :")
    for fn in ["model_correct.tsv", "model_accuracy.tsv", "category_accuracy.tsv", "model_x_category_accuracy.tsv"]:
        print(f"  analysis/{fn}")
    print("\nModel accuracy:")
    for n in models_sorted:
        print(f"  {n:28s} {model_c[n]:3d}/{model_t[n]:<3d}  {acc(model_c[n], model_t[n])}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
