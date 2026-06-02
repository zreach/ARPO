from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SEARCH_RE = re.compile(r"(<search>|</search>|\bsearch\b|wikipedia|bing|browser)", re.IGNORECASE)
CODE_RE = re.compile(r"(<python>|</python>|code_interpreter|python interpreter|python tool)", re.IGNORECASE)


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def as_messages(prompt: Any) -> list[dict[str, Any]]:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list):
        return [{"role": "unknown", "content": str(prompt)}]
    messages: list[dict[str, Any]] = []
    for message in prompt:
        if isinstance(message, dict):
            messages.append(message)
        else:
            messages.append({"role": "unknown", "content": str(message)})
    return messages


def prompt_text(prompt: Any, include_system: bool = True) -> str:
    messages = as_messages(prompt)
    if not include_system:
        messages = [message for message in messages if message.get("role") != "system"]
    return "\n".join(str(message.get("content", "")) for message in messages)


def quantiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    qs = values.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "min": float(qs.loc[0]),
        "p25": float(qs.loc[0.25]),
        "p50": float(qs.loc[0.5]),
        "p75": float(qs.loc[0.75]),
        "p90": float(qs.loc[0.9]),
        "p95": float(qs.loc[0.95]),
        "p99": float(qs.loc[0.99]),
        "max": float(qs.loc[1.0]),
        "mean": float(values.mean()),
    }


def value_counts(series: pd.Series, top_k: int) -> dict[str, int]:
    counts = series.astype(str).fillna("<NA>").value_counts(dropna=False).head(top_k)
    return {str(key): int(value) for key, value in counts.items()}


def extract_extra_info_stats(series: pd.Series, top_k: int) -> dict[str, Any]:
    tool_selection = Counter()
    tool_mode = Counter()
    need_tools_kwargs = Counter()
    parseable = 0

    for value in series:
        info = value
        if hasattr(info, "as_py"):
            info = info.as_py()
        if not isinstance(info, dict):
            continue
        parseable += 1
        if "tool_selection" in info:
            selection = info["tool_selection"]
            if isinstance(selection, (list, tuple)):
                tool_selection.update(str(item) for item in selection)
            else:
                tool_selection.update([str(selection)])
        if "tool_mode" in info:
            tool_mode.update([str(info["tool_mode"])])
        if "need_tools_kwargs" in info:
            need_tools_kwargs.update([str(info["need_tools_kwargs"])])

    return {
        "parseable_rows": parseable,
        "tool_selection_top": dict(tool_selection.most_common(top_k)),
        "tool_mode_top": dict(tool_mode.most_common(top_k)),
        "need_tools_kwargs_top": dict(need_tools_kwargs.most_common(top_k)),
    }


def build_stats(df: pd.DataFrame, prompt_key: str, top_k: int, sample_size: int) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "null_counts": {column: int(count) for column, count in df.isna().sum().items()},
    }

    for column in ("ability", "data_source", "split", "source"):
        if column in df.columns:
            stats[f"{column}_top"] = value_counts(df[column], top_k)

    if prompt_key in df.columns:
        prompt_all = df[prompt_key].apply(lambda prompt: prompt_text(prompt, include_system=True))
        prompt_no_system = df[prompt_key].apply(lambda prompt: prompt_text(prompt, include_system=False))
        messages = df[prompt_key].apply(as_messages)
        first_roles = messages.apply(lambda msgs: msgs[0].get("role", "<empty>") if msgs else "<empty>")
        role_counter = Counter()
        for item in messages:
            role_counter.update(str(message.get("role", "<missing>")) for message in item)

        stats["prompt"] = {
            "first_role_top": value_counts(first_roles, top_k),
            "role_counts": dict(role_counter.most_common(top_k)),
            "message_count_quantiles": quantiles(messages.apply(len)),
            "char_length_quantiles": quantiles(prompt_all.str.len()),
            "non_system_char_length_quantiles": quantiles(prompt_no_system.str.len()),
            "mentions_search_rows_all": int(prompt_all.apply(lambda text: bool(SEARCH_RE.search(text))).sum()),
            "mentions_search_rows_non_system": int(prompt_no_system.apply(lambda text: bool(SEARCH_RE.search(text))).sum()),
            "mentions_code_rows_all": int(prompt_all.apply(lambda text: bool(CODE_RE.search(text))).sum()),
            "mentions_code_rows_non_system": int(prompt_no_system.apply(lambda text: bool(CODE_RE.search(text))).sum()),
        }

    if "extra_info" in df.columns:
        stats["extra_info"] = extract_extra_info_stats(df["extra_info"], top_k)

    if "reward_model" in df.columns:
        stats["reward_model_type_top"] = value_counts(df["reward_model"].apply(lambda value: type(value).__name__), top_k)

    if sample_size > 0 and len(df) > 0:
        sample_df = df.head(sample_size)
        samples = []
        for idx, row in sample_df.iterrows():
            item = {"index": int(idx) if isinstance(idx, int) else str(idx)}
            for column in ("ability", "data_source", "source"):
                if column in row:
                    item[column] = to_builtin(row[column])
            if prompt_key in row:
                item["prompt_preview"] = prompt_text(row[prompt_key], include_system=True)[:1200]
            if "extra_info" in row:
                item["extra_info"] = to_builtin(row["extra_info"])
            samples.append(item)
        stats["samples"] = samples

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Print summary statistics for a parquet dataset.")
    parser.add_argument("parquet", nargs="+", help="One or more parquet files to inspect.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--json-out", default="", help="Optional path to write the JSON report.")
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for parquet in args.parquet:
        path = Path(parquet).expanduser()
        df = pd.read_parquet(path)
        report[str(path)] = build_stats(df, args.prompt_key, args.top_k, args.sample_size)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
