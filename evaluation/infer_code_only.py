#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(os.getcwd())

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer

from src.data_loader import DataLoader
from src.tools.python_tool import PythonTool
from src.utils import extract_answer


CODE_ONLY_PROMPT = """You are a helpful assistant that solves math problems step by step with access to a Python interpreter tool.
Given a question, first reason in <think> </think>, then provide the final answer in <answer> </answer>.
When calculation, enumeration, symbolic manipulation, or verification is useful, call Python by writing:
<python>
python code here
</python>
The system will return the result as:
<result> python interpreter result here </result>
You may call Python multiple times, but you must not use web search or any <search> tags.
In the final answer, put the exact answer inside \\boxed{}."""


PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r"</?search>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code-only tool inference for math evaluation.")
    parser.add_argument("--endpoints", type=str, nargs="+", required=True)
    parser.add_argument("--api_keys", type=str, nargs="+", default=None)
    parser.add_argument("--default_model", type=str, default="QwQ-32B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="aime24")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--turns", type=int, nargs="+", default=[1])
    parser.add_argument("--counts", type=int, default=1000000)
    parser.add_argument("--max_concurrent_requests", type=int, default=16)
    parser.add_argument("--sample_timeout", type=int, default=1800)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--conda_path", type=str, default="/opt/miniconda3")
    parser.add_argument("--conda_env", type=str, default="zhouyz")
    parser.add_argument("--python_max_concurrent", type=int, default=32)
    parser.add_argument("--python_timeout", type=int, default=120)
    parser.add_argument("--max_python_times", type=int, default=5)
    return parser.parse_args()


class CodeOnlyRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        api_keys = args.api_keys or ["EMPTY"] * len(args.endpoints)
        if len(api_keys) != len(args.endpoints):
            raise ValueError("len(api_keys) must match len(endpoints)")
        self.clients = [
            AsyncOpenAI(base_url=endpoint, api_key=api_key)
            for endpoint, api_key in zip(args.endpoints, api_keys)
        ]
        self.client_lock = asyncio.Lock()
        self.next_client = 0
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.python_tool = PythonTool(
            conda_path=args.conda_path,
            conda_env=args.conda_env,
            max_concurrent=args.python_max_concurrent,
        )

    async def get_client(self) -> AsyncOpenAI:
        async with self.client_lock:
            client = self.clients[self.next_client]
            self.next_client = (self.next_client + 1) % len(self.clients)
            return client

    def build_prompt(self, question: str, transcript: str) -> str:
        messages = [
            {"role": "system", "content": CODE_ONLY_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt + transcript

    async def call_llm(self, prompt: str, stop: list[str] | None) -> str:
        client = await self.get_client()
        response = await client.completions.create(
            model=self.args.default_model,
            prompt=prompt,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
            stop=stop,
            extra_body={
                "repetition_penalty": self.args.repetition_penalty,
                "include_stop_str_in_output": True,
            },
        )
        text = response.choices[0].text or ""
        if "<result>" in text:
            text = text.split("<result>")[0]
        if "</python>" in text:
            text = text.split("</python>")[0] + "</python>"
        if "</answer>" in text:
            text = text.split("</answer>")[0] + "</answer>"
        return text

    async def process_one(self, idx: int, question: str, answer: str) -> dict[str, Any]:
        started = time.time()
        session_id = hashlib.md5(f"{CODE_ONLY_PROMPT}_{question}".encode()).hexdigest()
        transcript = ""
        logs: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        llm_time = 0.0
        python_time = 0.0
        python_rounds = 0

        try:
            while True:
                prompt = self.build_prompt(question, transcript)
                llm_started = time.time()
                output = await self.call_llm(prompt, stop=["</python>", "</answer>"])
                llm_time += time.time() - llm_started
                transcript += output
                logs.append({"role": "assistant", "content": output})

                if SEARCH_RE.search(output):
                    warning = "<result>Search is disabled. Use only Python or provide the final answer.</result>"
                    transcript += warning
                    logs.append({"role": "tool", "name": "search_disabled", "content": warning})
                    continue

                python_match = PYTHON_RE.search(output)
                if python_match:
                    code = python_match.group(1).strip()
                    if python_rounds >= self.args.max_python_times:
                        result = "The maximum python call limit is exceeded. Provide the final answer without more tool calls."
                    else:
                        py_started = time.time()
                        result = await self.python_tool.execute(code, timeout=self.args.python_timeout)
                        elapsed = time.time() - py_started
                        python_time += elapsed
                        python_rounds += 1
                        tool_calls.append(
                            {
                                "tool": "python",
                                "code": code,
                                "result": result,
                                "elapsed": elapsed,
                            }
                        )
                    tool_text = f"<result>{result}</result>"
                    transcript += tool_text
                    logs.append({"role": "tool", "name": "python", "content": tool_text})
                    continue

                if "</answer>" not in output:
                    prompt = self.build_prompt(question, transcript)
                    llm_started = time.time()
                    tail = await self.call_llm(prompt, stop=None)
                    llm_time += time.time() - llm_started
                    transcript += tail
                    logs.append({"role": "assistant", "content": tail})
                break

            prediction = extract_answer(transcript)
            return {
                "idx": idx,
                "instruction": CODE_ONLY_PROMPT,
                "input": question,
                "output": transcript,
                "prediction": prediction,
                "answer": answer,
                "logs": logs,
                "tool_calls": tool_calls,
                "search_query_history": [],
                "timing": {
                    "llm_time": llm_time,
                    "python_time": python_time,
                    "search_time": 0.0,
                    "total_time": time.time() - started,
                },
                "session_id": session_id,
            }
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return {
                "idx": idx,
                "instruction": CODE_ONLY_PROMPT,
                "input": question,
                "output": transcript + f"\n[Error: {exc}]",
                "prediction": f"Error: {exc}",
                "answer": answer,
                "logs": logs,
                "tool_calls": tool_calls,
                "search_query_history": [],
                "timing": {
                    "llm_time": llm_time,
                    "python_time": python_time,
                    "search_time": 0.0,
                    "total_time": time.time() - started,
                },
                "session_id": session_id,
            }

    async def process_one_with_timeout(self, idx: int, question: str, answer: str) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self.process_one(idx, question, answer),
                timeout=self.args.sample_timeout,
            )
        except asyncio.TimeoutError:
            return {
                "idx": idx,
                "instruction": CODE_ONLY_PROMPT,
                "input": question,
                "output": f"Timeout after {self.args.sample_timeout}s",
                "prediction": "Timeout",
                "answer": answer,
                "logs": [],
                "tool_calls": [],
                "search_query_history": [],
                "timing": {"llm_time": 0.0, "python_time": 0.0, "search_time": 0.0, "total_time": self.args.sample_timeout},
            }

    async def run(self) -> None:
        loader = DataLoader(self.args.dataset_name, self.args.data_path)
        questions, answers = loader.load_data()
        total = min(len(questions), self.args.counts)
        questions = questions[:total]
        answers = answers[:total]

        for turn in self.args.turns:
            results: list[dict[str, Any] | None] = [None] * total
            queue: asyncio.Queue[int] = asyncio.Queue()
            for idx in range(total):
                await queue.put(idx)

            async def worker() -> None:
                while not queue.empty():
                    idx = await queue.get()
                    try:
                        results[idx] = await self.process_one_with_timeout(idx, questions[idx], answers[idx])
                    finally:
                        queue.task_done()

            workers = [
                asyncio.create_task(worker())
                for _ in range(min(self.args.max_concurrent_requests, total))
            ]
            pbar = async_tqdm(total=total, desc=f"{self.args.dataset_name} turn {turn}")
            done = 0
            while done < total:
                current = sum(result is not None for result in results)
                if current > done:
                    pbar.update(current - done)
                    done = current
                await asyncio.sleep(0.1)
            pbar.close()
            await queue.join()
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

            out_dir = Path(self.args.output_path) / self.args.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"{self.args.dataset_name}_output_{turn}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved trajectories to {output_file}")


async def main() -> None:
    args = parse_args()
    print(vars(args))
    runner = CodeOnlyRunner(args)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
