"""Safely repair v1/v2/v3/v4 aggregate usage stats.

Dry-run by default. Use --apply only after checking the printed summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any


AWARDS = ("O", "F", "M", "H", "S")
CURRENT_CALIBRATION_VERSION = "calibrated_v4_conservative_single"


def empty_counts() -> dict[str, int]:
    return {award: 0 for award in AWARDS}


def normalize_award(award: str) -> str:
    value = str(award or "S").upper().strip()
    if value in {"S/U", "U", "SUCCESSFUL PARTICIPANT"}:
        return "S"
    return value if value in AWARDS else "S"


def coerce_counts(value: Any) -> dict[str, int]:
    counts = empty_counts()
    if not isinstance(value, dict):
        return counts
    for key, raw in value.items():
        try:
            amount = int(float(raw))
        except (TypeError, ValueError):
            amount = 0
        counts[normalize_award(str(key))] += max(amount, 0)
    return counts


def merge_counts(*items: dict[str, int]) -> dict[str, int]:
    merged = empty_counts()
    for item in items:
        for award, count in coerce_counts(item).items():
            merged[award] += count
    return merged


def count_total(counts: dict[str, int]) -> int:
    return sum(coerce_counts(counts).values())


def find_counts(stats: dict[str, Any], version: str) -> dict[str, int]:
    version_counts = stats.get("version_award_counts")
    if isinstance(version_counts, dict) and isinstance(version_counts.get(version), dict):
        return coerce_counts(version_counts[version])
    if version == "v1":
        return coerce_counts(stats.get("legacy_award_counts"))
    if version == "v2":
        return coerce_counts(stats.get("current_version_award_counts"))
    if version == "v4":
        return coerce_counts(stats.get("current_version_award_counts"))
    return empty_counts()


def choose_counts(
    *,
    primary: dict[str, Any],
    source: dict[str, Any],
    version: str,
    expected_total: int,
    allow_placeholder: bool,
) -> dict[str, int]:
    candidates = [
        find_counts(source, version),
        find_counts(primary, version),
    ]
    if version == "v1":
        candidates.extend(
            [
                coerce_counts(source.get("legacy_award_counts")),
                coerce_counts(primary.get("legacy_award_counts")),
            ]
        )
    if version == "v2":
        candidates.extend(
            [
                coerce_counts(source.get("current_version_award_counts")),
                coerce_counts(primary.get("current_version_award_counts")),
            ]
        )
    if version == "v4":
        candidates.extend(
            [
                coerce_counts(source.get("current_version_award_counts")),
                coerce_counts(primary.get("current_version_award_counts")),
            ]
        )

    for counts in candidates:
        if count_total(counts) == expected_total:
            return counts

    if expected_total == 0:
        return empty_counts()

    if allow_placeholder:
        counts = empty_counts()
        counts["S"] = expected_total
        return counts

    totals = ", ".join(str(count_total(counts)) for counts in candidates)
    raise SystemExit(
        f"Cannot find {version} award counts summing to {expected_total}. "
        f"Candidate totals were: {totals}. "
        "Pass --source PATH pointing to the pre-v4 backup, or use --allow-placeholder "
        "only if you accept losing per-award distribution for that version."
    )


def stats_path_from_env() -> Path:
    base = Path(os.environ.get("RENDER_DISK_PATH", Path(__file__).resolve().parents[1] / "data"))
    return base / "usage_stats.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=stats_path_from_env())
    parser.add_argument("--source", type=Path, help="Optional pre-v4 backup JSON to source older counts from.")
    parser.add_argument("--v1-total", type=int, required=True)
    parser.add_argument("--v2-total", type=int, required=True)
    parser.add_argument("--v3-total", type=int, default=0)
    parser.add_argument("--v4-total", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-placeholder", action="store_true")
    args = parser.parse_args()

    primary = load_json(args.stats)
    source = load_json(args.source) if args.source else primary

    v1 = choose_counts(
        primary=primary,
        source=source,
        version="v1",
        expected_total=args.v1_total,
        allow_placeholder=args.allow_placeholder,
    )
    v2 = choose_counts(
        primary=primary,
        source=source,
        version="v2",
        expected_total=args.v2_total,
        allow_placeholder=args.allow_placeholder,
    )
    v3 = choose_counts(
        primary=primary,
        source=source,
        version="v3",
        expected_total=args.v3_total,
        allow_placeholder=args.allow_placeholder,
    )
    v4 = choose_counts(
        primary=primary,
        source=source,
        version="v4",
        expected_total=args.v4_total,
        allow_placeholder=args.allow_placeholder,
    )

    repaired = dict(primary)
    repaired["version_award_counts"] = {"v1": v1, "v2": v2, "v3": v3, "v4": v4}
    repaired["legacy_award_counts"] = merge_counts(v1, v2, v3)
    repaired["current_version_award_counts"] = v4
    repaired["award_counts"] = merge_counts(v1, v2, v3, v4)
    repaired["total_predictions"] = count_total(repaired["award_counts"])
    repaired["today_date"] = str(date.today())
    repaired["today_predictions"] = count_total(v4)
    if count_total(v4) == 0:
        repaired["current_version_recent_scores"] = []
        repaired["v4_recent_scores"] = []
    repaired["calibration_version"] = CURRENT_CALIBRATION_VERSION
    repaired["manual_repair"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "expected_totals": {
            "v1": args.v1_total,
            "v2": args.v2_total,
            "v3": args.v3_total,
            "v4": args.v4_total,
        },
        "source": str(args.source) if args.source else str(args.stats),
    }

    print("Repair summary")
    print(f"stats: {args.stats}")
    print(f"source: {args.source or args.stats}")
    for version, counts in repaired["version_award_counts"].items():
        print(f"{version}: total={count_total(counts)} counts={counts}")
    print(f"total_predictions={repaired['total_predictions']}")

    if not args.apply:
        print("DRY RUN ONLY. Re-run with --apply to write changes.")
        return

    backup_dir = args.stats.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"manual_repair_before_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(args.stats, backup_path)
    tmp_path = args.stats.with_suffix(f".repair.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, args.stats)
    print(f"WROTE repaired stats. Backup: {backup_path}")


if __name__ == "__main__":
    main()
