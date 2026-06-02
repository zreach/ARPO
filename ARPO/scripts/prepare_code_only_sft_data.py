from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_messages(row: dict[str, Any], system_prompt: str | None) -> list[dict[str, str]]:
    instruction = system_prompt if system_prompt is not None else str(row.get("instruction", ""))
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": str(row.get("input", ""))},
        {"role": "assistant", "content": str(row.get("output", ""))},
    ]


def is_code_only(row: dict[str, Any]) -> bool:
    text = "\n".join(str(row.get(key, "")) for key in ("instruction", "input", "output")).lower()
    return "<python>" in text and "<search>" not in text and "</search>" not in text


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare code-only SFT parquet data.")
    parser.add_argument("--source", required=True, help="Code-only SFT JSONL/parquet source.")
    parser.add_argument("--train", required=True, help="Output train parquet.")
    parser.add_argument("--val", required=True, help="Output validation parquet.")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt override.")
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--summary", default="", help="Optional summary JSON path.")
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    rows = [row for row in read_rows(source) if is_code_only(row)]
    if not rows:
        raise ValueError(f"No code-only rows found in {source}")

    system_prompt = None
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).expanduser().read_text(encoding="utf-8").strip()

    data = [{"messages": to_messages(row, system_prompt), "extra_info": row.get("extra_info", {})} for row in rows]
    val_size = min(max(args.val_size, 0), len(data))
    val_rows = data[:val_size]
    train_rows = data[val_size:] if val_size else data

    train_path = Path(args.train).expanduser()
    val_path = Path(args.val).expanduser()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(val_rows or train_rows[:1]).to_parquet(val_path)

    summary = {
        "source": str(source),
        "total_code_only_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows or train_rows[:1]),
        "train": str(train_path),
        "val": str(val_path),
    }
    if args.summary:
        summary_path = Path(args.summary).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(val_rows or train_rows[:1])} val rows to {val_path}")


if __name__ == "__main__":
    main()
