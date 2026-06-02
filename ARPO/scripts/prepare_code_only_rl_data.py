from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEARCH_MARKERS = (
    "<search>",
    "</search>",
    "search tool",
    "web search",
    "bingsearch",
    "bing search",
    "wikipedia",
    "browser",
)


def as_messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")
    return [dict(message) if isinstance(message, dict) else {"role": "user", "content": str(message)} for message in prompt]


def prompt_text(prompt: Any, include_system: bool = True) -> str:
    try:
        messages = as_messages(prompt)
    except TypeError:
        return str(prompt)
    if not include_system:
        messages = [message for message in messages if message.get("role") != "system"]
    return "\n".join(str(message.get("content", "")) for message in messages)


def replace_system_prompt(prompt: Any, system_prompt: str) -> list[dict[str, Any]]:
    messages = as_messages(prompt)
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": system_prompt}
        return messages
    return [{"role": "system", "content": system_prompt}] + messages


def patch_extra_info(extra_info: Any) -> dict[str, Any]:
    extra = dict(extra_info) if isinstance(extra_info, dict) else {}
    extra["need_tools_kwargs"] = False
    extra["tool_selection"] = ["code_interpreter"]
    extra["tool_mode"] = "code_only"
    return extra


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip().lower()


def looks_like_search_row(row: pd.Series, prompt_key: str) -> bool:
    # The original ARPO system prompt can mention search for every row. Ignore
    # that old system message because it is replaced with a pure code prompt.
    text = prompt_text(row[prompt_key], include_system=False).lower()
    return any(marker in text for marker in SEARCH_MARKERS)


def keep_code_only_row(row: pd.Series, prompt_key: str, ability: str) -> bool:
    if looks_like_search_row(row, prompt_key):
        return False
    if "ability" in row:
        return normalize_value(row["ability"]) == ability.lower()
    return True


def filter_frame(df: pd.DataFrame, prompt_key: str, ability: str, system_prompt: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if prompt_key not in df.columns:
        raise KeyError(f"Missing prompt column {prompt_key!r}; available columns: {list(df.columns)}")

    before = len(df)
    search_mask = df.apply(lambda row: looks_like_search_row(row, prompt_key), axis=1)
    if "ability" in df.columns:
        ability_mask = df["ability"].apply(lambda value: normalize_value(value) == ability.lower())
    else:
        ability_mask = pd.Series([True] * len(df), index=df.index)
    keep_mask = df.apply(lambda row: keep_code_only_row(row, prompt_key, ability), axis=1)
    kept = df[keep_mask].copy()
    dropped = before - len(kept)

    kept[prompt_key] = kept[prompt_key].apply(lambda prompt: replace_system_prompt(prompt, system_prompt))
    if "extra_info" in kept.columns:
        kept["extra_info"] = kept["extra_info"].apply(patch_extra_info)

    return kept, {
        "input_rows": before,
        "kept_rows": len(kept),
        "dropped_rows": dropped,
        "ability_matched_rows": int(ability_mask.sum()),
        "user_prompt_search_rows": int(search_mask.sum()),
    }


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ARPO code-only RL dataset from mixed ARPO parquet files.")
    parser.add_argument("--train-source", required=True, help="Source train parquet, usually rl_datasets/train_10k.parquet.")
    parser.add_argument("--train-target", required=True, help="Output code-only train parquet.")
    parser.add_argument("--val-source", default="", help="Optional source validation parquet.")
    parser.add_argument("--val-target", required=True, help="Output code-only validation parquet.")
    parser.add_argument("--system-prompt", required=True, help="Pure code-tool system prompt.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--ability", default="math", help="Ability value to keep from mixed ARPO data.")
    parser.add_argument("--fallback-val-size", type=int, default=256)
    parser.add_argument("--summary", default="", help="Optional output summary JSON.")
    args = parser.parse_args()

    system_prompt = Path(args.system_prompt).expanduser().read_text(encoding="utf-8").strip()
    if re.search(r"\bsearch\b", system_prompt, flags=re.IGNORECASE):
        raise ValueError(f"System prompt still mentions search: {args.system_prompt}")

    train_df = pd.read_parquet(Path(args.train_source).expanduser())
    train_code_only, train_stats = filter_frame(train_df, args.prompt_key, args.ability, system_prompt)

    if len(train_code_only) == 0:
        ability_counts = (
            train_df["ability"].astype(str).value_counts().head(20).to_dict()
            if "ability" in train_df.columns
            else {}
        )
        raise ValueError(
            "No code-only train rows remained after filtering. "
            f"stats={train_stats}, ability_counts={ability_counts}"
        )

    val_stats: dict[str, int]
    val_source = Path(args.val_source).expanduser() if args.val_source else None
    if val_source and val_source.exists():
        val_df = pd.read_parquet(val_source)
        val_code_only, val_stats = filter_frame(val_df, args.prompt_key, args.ability, system_prompt)
    else:
        val_code_only = train_code_only.head(max(args.fallback_val_size, 1)).copy()
        val_stats = {
            "input_rows": 0,
            "kept_rows": len(val_code_only),
            "dropped_rows": 0,
        }

    if len(val_code_only) == 0:
        val_size = min(max(args.fallback_val_size, 1), len(train_code_only))
        val_code_only = train_code_only.head(val_size).copy()
        val_stats["fallback_from_train_rows"] = len(val_code_only)

    train_target = Path(args.train_target).expanduser()
    val_target = Path(args.val_target).expanduser()
    write_parquet(train_code_only, train_target)
    write_parquet(val_code_only, val_target)

    summary = {
        "train_source": str(Path(args.train_source).expanduser()),
        "train_target": str(train_target),
        "val_source": str(val_source) if val_source else "",
        "val_target": str(val_target),
        "prompt_key": args.prompt_key,
        "ability": args.ability,
        "train": train_stats,
        "val": val_stats,
    }
    summary_path = Path(args.summary).expanduser() if args.summary else train_target.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(train_code_only)} code-only train rows to {train_target}")
    print(f"Wrote {len(val_code_only)} code-only val rows to {val_target}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
