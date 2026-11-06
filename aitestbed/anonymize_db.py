#!/usr/bin/env python3
"""
Anonymize provider/model fields in the results database.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from analysis.anonymization import get_anonymizer


def anonymize_metadata(anonymizer, metadata: str) -> str:
    if not metadata:
        return metadata
    try:
        data = json.loads(metadata)
    except Exception:
        return metadata

    changed = False
    for key in ("model", "tts_model"):
        if key in data and isinstance(data[key], str):
            alias = anonymizer.model_alias(data[key])
            if alias != data[key]:
                data[key] = alias
                changed = True
    if "provider" in data and isinstance(data["provider"], str):
        alias = anonymizer.provider_alias(data["provider"])
        if alias != data["provider"]:
            data["provider"] = alias
            changed = True

    return json.dumps(data) if changed else metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymize provider/model fields in traffic_logs.db")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Database path")
    parser.add_argument("--map", default=None, help="Path to anonymization_map.json")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    anonymizer = get_anonymizer(map_path=args.map)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, provider, model, metadata FROM traffic_logs")
    rows = cur.fetchall()

    updated = 0
    for row_id, provider, model, metadata in rows:
        new_provider = anonymizer.provider_alias(provider)
        new_model = anonymizer.model_alias(model)
        new_metadata = anonymize_metadata(anonymizer, metadata)

        if new_provider != provider or new_model != model or new_metadata != metadata:
            updated += 1
            if not args.dry_run:
                cur.execute(
                    "UPDATE traffic_logs SET provider = ?, model = ?, metadata = ? WHERE id = ?",
                    (new_provider, new_model, new_metadata, row_id),
                )

    if not args.dry_run:
        conn.commit()
    conn.close()

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: {updated} rows processed")


if __name__ == "__main__":
    main()
