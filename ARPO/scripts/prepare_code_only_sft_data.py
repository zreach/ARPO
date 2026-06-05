from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEARCH_RE = re.compile(
    r"(<search>|</search>|search|web\s*search|bing|google|wikipedia|browser|retrieval)",
    re.IGNORECASE,
)


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


def row_text(row: dict[str, Any], include_instruction: bool) -> str:
    keys = ["input", "output"]
    if include_instruction:
        keys.insert(0, "instruction")
    return "\n".join(str(row.get(key, "")) for key in keys)


def has_search_text(text: str) -> bool:
    return bool(SEARCH_RE.search(text))


def is_code_only(row: dict[str, Any], include_instruction: bool) -> bool:
    text = row_text(row, include_instruction=include_instruction)
    return "<python>" in text.lower() and not has_search_text(text)


def sanitize_extra_info(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if has_search_text(str(key)):
                continue
            sanitized = sanitize_extra_info(item)
            if sanitized is not None:
                cleaned[key] = sanitized
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            sanitized = sanitize_extra_info(item)
            if sanitized is not None:
                cleaned_list.append(sanitized)
        return cleaned_list
    if isinstance(value, str) and has_search_text(value):
        return None
    return value


def validate_messages(messages: list[dict[str, str]]) -> None:
    text = "\n".join(message["content"] for message in messages)
    if has_search_text(text):
        raise ValueError("Prepared SFT messages still contain search-related text.")


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
    system_prompt = None
    if args.system_prompt:
        system_prompt = Path(args.system_prompt).expanduser().read_text(encoding="utf-8").strip()
        if has_search_text(system_prompt):
            raise ValueError(f"System prompt still contains search-related text: {args.system_prompt}")

    raw_rows = read_rows(source)
    include_instruction = system_prompt is None
    rows = [row for row in raw_rows if is_code_only(row, include_instruction=include_instruction)]
    if not rows:
        raise ValueError(f"No code-only rows found in {source}")

    data = []
    for row in rows:
        messages = to_messages(row, system_prompt)
        validate_messages(messages)
        data.append({"messages": messages, "extra_info": sanitize_extra_info(row.get("extra_info", {}))})
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
        "input_rows": len(raw_rows),
        "dropped_search_or_non_code_rows": len(raw_rows) - len(rows),
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
