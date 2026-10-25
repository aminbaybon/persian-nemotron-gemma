"""Score one local model on a frozen Berkeley Function-Calling Leaderboard slice.

The slice is BFCL v4 non-live Python: simple, multiple, parallel, and irrelevance.
Questions and possible answers are frozen in data/bfcl_sample.json.
This is a single-turn AST check. Tools are not executed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "bfcl_sample.json"
RAW_DIR = ROOT / "data" / "bfcl"
ANSWER_DIR = RAW_DIR / "possible_answer"
RESULTS = ROOT / "results"

SEED = 25
PER_CATEGORY = 12
CATEGORIES = (
    ("simple_python", "BFCL_v4_simple_python.json", True),
    ("multiple", "BFCL_v4_multiple.json", True),
    ("parallel", "BFCL_v4_parallel.json", True),
    ("irrelevance", "BFCL_v4_irrelevance.json", False),
)

PUNCT_RE = re.compile(r"[ ,.\/\-\_\*\^]")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_sample() -> list[dict]:
    rng = random.Random(SEED)
    sample = []
    for category, filename, has_answer in CATEGORIES:
        questions = {row["id"]: row for row in load_jsonl(RAW_DIR / filename)}
        answers = {}
        if has_answer:
            for row in load_jsonl(ANSWER_DIR / filename):
                answers[row["id"]] = row["ground_truth"]
        ids = sorted(questions)
        rng.shuffle(ids)
        for item_id in ids[:PER_CATEGORY]:
            question = questions[item_id]
            sample.append(
                {
                    "id": item_id,
                    "category": category,
                    "question": question["question"][0][0]["content"],
                    "functions": question["function"],
                    "ground_truth": answers.get(item_id, []),
                }
            )
    return sample


def standardize_string(value: str) -> str:
    return PUNCT_RE.sub("", value).lower().replace("'", '"')


def openai_tool(function: dict) -> dict:
    parameters = json.loads(json.dumps(function.get("parameters") or {}))
    if parameters.get("type") == "dict":
        parameters["type"] = "object"
    for prop in (parameters.get("properties") or {}).values():
        if isinstance(prop, dict):
            prop.pop("optional", None)
            if prop.get("type") == "dict":
                prop["type"] = "object"
            if prop.get("type") == "float":
                prop["type"] = "number"
            if prop.get("type") == "tuple":
                prop["type"] = "array"
    return {
        "type": "function",
        "function": {
            "name": function["name"],
            "description": function.get("description") or "",
            "parameters": parameters,
        },
    }


def function_index(functions: list[dict]) -> dict[str, dict]:
    return {function["name"]: function for function in functions}


def concrete_options(options: list) -> list:
    return [option for option in options if option != ""]


def names_match(expected: str, actual: str) -> bool:
    if actual == expected:
        return True
    return actual.replace(".", "_") == expected.replace(".", "_")


def string_ok(actual: str, options: list) -> bool:
    standardized = standardize_string(actual)
    allowed = [standardize_string(option) for option in options if isinstance(option, str)]
    return standardized in allowed


def number_ok(actual, options: list) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    for option in options:
        if isinstance(option, bool) or not isinstance(option, (int, float)):
            continue
        if abs(float(actual) - float(option)) < 1e-6:
            return True
    return False


def list_ok(actual: list, options: list) -> bool:
    normalized = [standardize_string(item) if isinstance(item, str) else item for item in actual]
    for option in options:
        if not isinstance(option, list) or len(option) != len(normalized):
            continue
        candidate = [standardize_string(item) if isinstance(item, str) else item for item in option]
        if candidate == normalized:
            return True
    return False


def dict_ok(actual: dict, options: list) -> bool:
    for option in options:
        if option == "" or not isinstance(option, dict):
            continue
        if set(actual) - set(option):
            continue
        matched = True
        for key, value in actual.items():
            allowed = option[key]
            standardized = standardize_string(value) if isinstance(value, str) else value
            pool = [standardize_string(item) if isinstance(item, str) else item for item in allowed]
            if standardized not in pool:
                matched = False
                break
        if not matched:
            continue
        for key, allowed in option.items():
            if key not in actual and "" not in allowed:
                matched = False
                break
        if matched:
            return True
    return False


def value_ok(actual, options: list, expected_type: str) -> bool:
    concrete = concrete_options(options)
    if not concrete:
        return False
    if expected_type == "string":
        return isinstance(actual, str) and string_ok(actual, concrete)
    if expected_type in {"integer", "float"}:
        return number_ok(actual, concrete)
    if expected_type == "boolean":
        return isinstance(actual, bool) and actual in concrete
    if expected_type in {"array", "tuple"}:
        return isinstance(actual, list) and list_ok(actual, concrete)
    if expected_type == "dict":
        return isinstance(actual, dict) and dict_ok(actual, concrete)
    return actual in concrete


def call_matches(gold: dict, pred: dict, functions: dict[str, dict]) -> tuple[bool, str]:
    if len(gold) != 1:
        return False, "bad_gold"
    expected_name, expected_args = next(iter(gold.items()))
    if not names_match(expected_name, pred.get("name") or ""):
        return False, "wrong_function"
    schema = functions.get(expected_name) or functions.get(pred.get("name") or "")
    if schema is None:
        for function in functions.values():
            if names_match(function["name"], pred.get("name") or ""):
                schema = function
                break
    if schema is None:
        return False, "unknown_function"
    args = pred.get("arguments")
    if not isinstance(args, dict):
        return False, "unparsed_arguments"
    properties = (schema.get("parameters") or {}).get("properties") or {}
    required = (schema.get("parameters") or {}).get("required") or []
    for name in required:
        if name not in args:
            return False, f"missing_required:{name}"
    for name, value in args.items():
        if name not in properties or name not in expected_args:
            return False, f"unexpected_param:{name}"
        expected_type = properties[name].get("type", "string")
        if expected_type == "number":
            expected_type = "float"
        if not value_ok(value, expected_args[name], expected_type):
            return False, f"bad_value:{name}"
    for name, options in expected_args.items():
        if name not in args and "" not in options:
            return False, f"missing_param:{name}"
    return True, ""


def score_calls(task: dict, calls: list[dict]) -> dict:
    functions = function_index(task["functions"])
    if task["category"] == "irrelevance":
        return {
            "success": len(calls) == 0,
            "incorrect_tool_calls": len(calls),
            "reasons": ["called_tool"] if calls else [],
        }
    golds = list(task["ground_truth"])
    used = [False] * len(golds)
    incorrect = 0
    reasons = []
    for call in calls:
        matched = False
        last_reason = ""
        for index, gold in enumerate(golds):
            if used[index]:
                continue
            ok, reason = call_matches(gold, call, functions)
            if ok:
                used[index] = True
                matched = True
                break
            last_reason = reason
        if not matched:
            incorrect += 1
            if last_reason:
                reasons.append(last_reason)
    missing = sum(1 for flag in used if not flag)
    if missing:
        reasons.append(f"missing_calls:{missing}")
    return {
        "success": incorrect == 0 and missing == 0 and len(calls) == len(golds),
        "incorrect_tool_calls": incorrect,
        "reasons": reasons,
    }


def post_json(base_url: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc


def parse_arguments(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_calls(message: dict) -> list[dict]:
    calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        calls.append(
            {
                "name": function.get("name") or "",
                "arguments": parse_arguments(function.get("arguments")),
            }
        )
    return calls


def run_task(base_url: str, task: dict) -> dict:
    body = {
        "messages": [{"role": "user", "content": task["question"]}],
        "tools": [openai_tool(function) for function in task["functions"]],
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    error = ""
    calls: list[dict] = []
    content = ""
    prompt_tokens = 0
    completion_tokens = 0
    try:
        payload = post_json(base_url, body)
        message = payload["choices"][0]["message"] or {}
        content = message.get("content") or ""
        calls = extract_calls(message)
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except Exception as exc:  # noqa: BLE001 - keep the rest of the sample
        error = str(exc)
    latency = time.perf_counter() - started
    verdict = score_calls(task, calls) if not error else {
        "success": False,
        "incorrect_tool_calls": 0,
        "reasons": ["http_error"],
    }
    return {
        "id": task["id"],
        "category": task["category"],
        "success": verdict["success"],
        "incorrect_tool_calls": verdict["incorrect_tool_calls"],
        "reasons": verdict["reasons"],
        "latency_seconds": round(latency, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_consumed": prompt_tokens + completion_tokens,
        "calls": calls,
        "content": content,
        "error": error,
    }


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    successes = sum(1 for row in rows if row["success"])
    latency = sum(row["latency_seconds"] for row in rows)
    by_category = {}
    for category, _, _ in CATEGORIES:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "successes": sum(1 for row in subset if row["success"]),
            "success_rate": round(
                sum(1 for row in subset if row["success"]) / len(subset), 4
            )
            if subset
            else 0.0,
            "incorrect_tool_calls": sum(row["incorrect_tool_calls"] for row in subset),
        }
    return {
        "n_tasks": total,
        "task_successes": successes,
        "task_success_rate": round(successes / total, 4) if total else 0.0,
        "incorrect_tool_calls": sum(row["incorrect_tool_calls"] for row in rows),
        "e2e_latency_seconds": round(latency, 3),
        "e2e_latency_mean_seconds": round(latency / total, 3) if total else 0.0,
        "tokens_consumed": sum(row["tokens_consumed"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "by_category": by_category,
    }


def self_test() -> None:
    task = {
        "id": "simple_python_0",
        "category": "simple_python",
        "functions": [
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
        "ground_truth": [
            {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
        ],
    }
    good = score_calls(
        task,
        [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}],
    )
    if not good["success"] or good["incorrect_tool_calls"] != 0:
        raise SystemExit(f"self-test good failed: {good}")
    bad = score_calls(
        task,
        [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 4}}],
    )
    if bad["success"] or bad["incorrect_tool_calls"] != 1:
        raise SystemExit(f"self-test bad failed: {bad}")
    extra = score_calls(
        task,
        [
            {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "unit": "units"}},
            {"name": "calculate_triangle_area", "arguments": {"base": 1, "height": 1}},
        ],
    )
    if extra["success"] or extra["incorrect_tool_calls"] != 1:
        raise SystemExit(f"self-test extra failed: {extra}")
    irrelevant = score_calls(
        {"category": "irrelevance", "functions": [], "ground_truth": []},
        [{"name": "determine_body_mass_index", "arguments": {}}],
    )
    if irrelevant["success"] or irrelevant["incorrect_tool_calls"] != 1:
        raise SystemExit(f"self-test irrelevance failed: {irrelevant}")
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model-name", default="prepare")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated BFCL ids")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.prepare_only or not SAMPLE_PATH.exists():
        sample = build_sample()
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {SAMPLE_PATH} n={len(sample)}")
        if args.prepare_only:
            return
    else:
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        sample = [item for item in sample if item["id"] in wanted]
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, task in enumerate(sample, start=1):
        row = run_task(args.base_url, task)
        rows.append(row)
        print(
            f"{args.model_name} {task['category']} {task['id']} {index}/{len(sample)} "
            f"success={int(row['success'])} incorrect={row['incorrect_tool_calls']} "
            f"{row['latency_seconds']}s tokens={row['tokens_consumed']}",
            flush=True,
        )
    report = {"model": args.model_name, "base_url": args.base_url, "summary": aggregate(rows), "rows": rows}
    out_path = RESULTS / f"bfcl-{args.model_name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
