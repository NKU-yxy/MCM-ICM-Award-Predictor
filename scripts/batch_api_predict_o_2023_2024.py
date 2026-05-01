"""Batch score 2023/2024 O-award PDFs through the running local web API."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
OUT_DIR = Path(os.getenv("BATCH_OUT_DIR", r"C:\tmp\mcm_batch_results"))
API_URL = os.getenv("BATCH_API_URL", "http://127.0.0.1:8000/api/predict")
YEARS = (2023, 2024)


def iter_pdfs() -> list[Path]:
    pdfs: list[Path] = []
    for year in YEARS:
        pdfs.extend(sorted((DATA_ROOT / str(year)).rglob("*_O.pdf")))
    return pdfs


def infer_problem(path: Path) -> str:
    folder = path.parent.name.upper()
    if "_" in folder:
        return folder.rsplit("_", 1)[-1]
    return "auto"


def compact_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rubric = payload.get("llm_rubric") or {}
    details = rubric.get("details") or {}
    probabilities = rubric.get("probabilities") or {}
    metadata = payload.get("metadata") or {}
    structure = payload.get("structure") or {}
    rel = path.relative_to(ROOT).as_posix()
    return {
        "file": rel,
        "year": payload.get("year"),
        "contest": payload.get("contest"),
        "problem": payload.get("problem"),
        "status": rubric.get("status"),
        "score": rubric.get("score"),
        "award_prediction": rubric.get("award_prediction"),
        "p_O": probabilities.get("O"),
        "p_F": probabilities.get("F"),
        "p_M": probabilities.get("M"),
        "p_H": probabilities.get("H"),
        "p_S": probabilities.get("S"),
        "content_score": details.get("content_score"),
        "length_style_score": details.get("length_style_score"),
        "visual_score": details.get("visual_score"),
        "conclusion_score": details.get("conclusion_score"),
        "writing_score": details.get("writing_score"),
        "abstract_words": metadata.get("abstract_word_count"),
        "pages": metadata.get("page_count"),
        "images": metadata.get("image_count"),
        "figure_captions": structure.get("figure_caption_count"),
        "tables": structure.get("table_count"),
        "references": metadata.get("ref_count"),
        "comments": rubric.get("comments"),
    }


def post_pdf(path: Path) -> dict[str, Any]:
    problem = infer_problem(path)
    year = path.parts[path.parts.index("raw") + 1]
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, "application/pdf")}
        data = {"problem": problem, "year": year}
        response = requests.post(API_URL, files=files, data=data, timeout=180)
    if response.status_code == 429:
        raise RuntimeError("rate_limited")
    response.raise_for_status()
    return response.json()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = iter_pdfs()
    limit = int(os.getenv("BATCH_LIMIT", "0") or "0")
    if limit > 0:
        pdfs = pdfs[:limit]

    results: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []

    for idx, path in enumerate(pdfs, start=1):
        print(f"[{idx}/{len(pdfs)}] {path.relative_to(ROOT)}", flush=True)
        attempts = 0
        while True:
            attempts += 1
            try:
                payload = post_pdf(path)
                raw_results.append({"file": str(path.relative_to(ROOT)), "response": payload})
                row = compact_result(path, payload)
                results.append(row)
                print(
                    f"  -> {row['status']} score={row['score']} award={row['award_prediction']} "
                    f"prob=O{row['p_O']}/F{row['p_F']}/M{row['p_M']}/H{row['p_H']}/S{row['p_S']}",
                    flush=True,
                )
                break
            except RuntimeError as exc:
                if str(exc) != "rate_limited" or attempts > 8:
                    raise
                wait = 65
                print(f"  -> rate limited, sleep {wait}s", flush=True)
                time.sleep(wait)
            except Exception as exc:
                results.append({
                    "file": path.relative_to(ROOT).as_posix(),
                    "year": path.parts[path.parts.index("raw") + 1],
                    "contest": path.parent.name.split("_")[0],
                    "problem": infer_problem(path),
                    "status": "failed",
                    "score": None,
                    "award_prediction": None,
                    "comments": str(exc),
                })
                print(f"  -> failed: {exc}", flush=True)
                break

        time.sleep(float(os.getenv("BATCH_SLEEP", "1.0")))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"o_2023_2024_api_{timestamp}.json"
    csv_path = OUT_DIR / f"o_2023_2024_api_{timestamp}.csv"
    json_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "file", "year", "contest", "problem", "status", "score", "award_prediction",
        "p_O", "p_F", "p_M", "p_H", "p_S",
        "content_score", "length_style_score", "visual_score", "conclusion_score", "writing_score",
        "abstract_words", "pages", "images", "figure_captions", "tables", "references", "comments",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
