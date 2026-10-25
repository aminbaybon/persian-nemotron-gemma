"""Score one local model on a frozen simulated tool-calling suite.

The world, tool schemas, prompts, and success rules live in data/tool_tasks.json.
Both models must be scored with that file unchanged. Tools are in-process fakes.
No network call reaches a bank or a device.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "data" / "tool_tasks.json"
RESULTS = ROOT / "results"

DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

CHECKPOINTS = {
    "nemotron-ud-q4-k-xl": {
        "file_name": "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf",
        "quant": "UD-Q4_K_XL",
        "path": Path(r"D:\Api-test\nemotron\models\NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf"),
    },
    "gemma-4-e4b": {
        "file_name": "gemma-4-E4B-it-Q5_K_M.gguf",
        "quant": "Q5_K_M",
        "path": Path(r"D:\Api-test\quantized model\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q5_K_M.gguf"),
    },
}


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def fresh_world(spec: dict) -> dict:
    return copy.deepcopy(spec["world"])


def parse_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().translate(DIGIT_TRANSLATION)
        text = text.replace(",", "").replace("٬", "").replace("،", "").replace("_", "")
        text = re.sub(r"(?<=\d)[ ](?=\d{3}(?:\D|$))", "", text)
        text = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", text)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text)
    return None


def values_equal(expected, actual) -> bool:
    left = parse_number(expected)
    right = parse_number(actual)
    if left is not None and right is not None:
        return abs(left - right) < 1e-6
    return str(expected) == str(actual)


def args_match(rule: dict, args: dict) -> bool:
    expected = rule.get("args") or {}
    if not isinstance(args, dict):
        return False
    if rule.get("ignore_extra"):
        if not set(expected).issubset(args):
            return False
    elif set(args) != set(expected):
        return False
    return all(values_equal(value, args.get(key)) for key, value in expected.items())


def classify_call(task: dict, counts: dict, name: str, args) -> str:
    """Return ok, incorrect, or forbidden. Forbidden calls are also incorrect."""
    parsed = args if isinstance(args, dict) else None
    if parsed is None:
        allows_tool = any(rule["tool"] == name for rule in task["allow"])
        forbids_tool = any(rule["tool"] == name and "args" not in rule for rule in task["forbidden"])
        if forbids_tool and not allows_tool:
            return "forbidden"
        return "incorrect"
    for index, rule in enumerate(task["allow"]):
        if rule["tool"] != name or not args_match(rule, parsed):
            continue
        key = ("allow", index)
        used = counts.get(key, 0)
        counts[key] = used + 1
        if used < int(rule.get("max_times", 1)):
            return "ok"
        return rule.get("on_excess", "incorrect")
    for rule in task["forbidden"]:
        if rule["tool"] == name and ("args" not in rule or args_match(rule, parsed)):
            return "forbidden"
    return "incorrect"


def tool_error(code: str, **extra) -> dict:
    body = {"ok": False, "error": code}
    body.update(extra)
    return body


def dispatch(world: dict, name: str, args: dict) -> dict:
    if name == "get_balance":
        account_id = str(args.get("account_id", ""))
        account = world["accounts"].get(account_id)
        if account is None:
            return tool_error("unknown_account", account_id=account_id)
        return {"ok": True, "account_id": account_id, "balance_irr": account["balance_irr"]}
    if name == "list_recent_transactions":
        account_id = str(args.get("account_id", ""))
        if account_id not in world["accounts"]:
            return tool_error("unknown_account", account_id=account_id)
        limit = parse_number(args.get("limit"))
        if limit is None or not limit.is_integer() or not 1 <= int(limit) <= 10:
            return tool_error("invalid_arguments", detail="limit must be an integer from 1 to 10")
        rows = world["transactions"].get(account_id, [])[: int(limit)]
        return {"ok": True, "account_id": account_id, "transactions": rows}
    if name == "transfer":
        from_id = str(args.get("from_id", ""))
        to_id = str(args.get("to_id", ""))
        if from_id not in world["accounts"] or to_id not in world["accounts"]:
            return tool_error("unknown_account", from_id=from_id, to_id=to_id)
        if from_id == to_id:
            return tool_error("invalid_arguments", detail="from_id and to_id must differ")
        amount = parse_number(args.get("amount_irr"))
        if amount is None or not amount.is_integer() or amount <= 0:
            return tool_error("invalid_arguments", detail="amount_irr must be a positive integer")
        amount_irr = int(amount)
        balance = world["accounts"][from_id]["balance_irr"]
        if balance < amount_irr:
            return tool_error("insufficient_funds", balance_irr=balance, amount_irr=amount_irr)
        world["accounts"][from_id]["balance_irr"] = balance - amount_irr
        world["accounts"][to_id]["balance_irr"] += amount_irr
        return {
            "ok": True,
            "from_id": from_id,
            "to_id": to_id,
            "amount_irr": amount_irr,
            "from_balance_irr": world["accounts"][from_id]["balance_irr"],
        }
    if name == "get_card_status":
        card_id = str(args.get("card_id", ""))
        card = world["cards"].get(card_id)
        if card is None:
            return tool_error("unknown_card", card_id=card_id)
        return {"ok": True, "card_id": card_id, "status": card["status"], "account_id": card["account_id"]}
    if name == "read_sensor":
        device_id = str(args.get("device_id", ""))
        channel = str(args.get("channel", ""))
        device = world["devices"].get(device_id)
        if device is None:
            return tool_error("unknown_device", device_id=device_id)
        if channel not in device["sensors"]:
            return tool_error("unknown_channel", channel=channel)
        return {"ok": True, "device_id": device_id, "channel": channel, "value": device["sensors"][channel]}
    if name == "get_alarm_limits":
        device_id = str(args.get("device_id", ""))
        device = world["devices"].get(device_id)
        if device is None:
            return tool_error("unknown_device", device_id=device_id)
        return {
            "ok": True,
            "device_id": device_id,
            "limits": device["alarm_limits"],
            "published_safe_bounds": device["published_safe_bounds"],
        }
    if name == "set_alarm_limit":
        device_id = str(args.get("device_id", ""))
        channel = str(args.get("channel", ""))
        device = world["devices"].get(device_id)
        if device is None:
            return tool_error("unknown_device", device_id=device_id)
        if channel not in device["alarm_limits"]:
            return tool_error("unknown_channel", channel=channel)
        low = parse_number(args.get("low"))
        high = parse_number(args.get("high"))
        if low is None or high is None:
            return tool_error("invalid_arguments", detail="low and high must be numbers")
        bounds = device["published_safe_bounds"][channel]
        reasons = []
        if not bounds["low"][0] <= low <= bounds["low"][1]:
            reasons.append("low_outside_published_safe_range")
        if not bounds["high"][0] <= high <= bounds["high"][1]:
            reasons.append("high_outside_published_safe_range")
        if low >= high:
            reasons.append("low_must_be_below_high")
        if reasons:
            return {
                "ok": False,
                "error": "rejected",
                "applied": False,
                "reasons": reasons,
                "published_safe_bounds": bounds,
                "current": device["alarm_limits"][channel],
            }
        device["alarm_limits"][channel] = {"low": low, "high": high}
        return {
            "ok": True,
            "applied": True,
            "device_id": device_id,
            "channel": channel,
            "limits": device["alarm_limits"][channel],
        }
    return tool_error("unknown_tool", tool=name)


def lookup_path(world: dict, path: str):
    current = world
    for part in path.split("."):
        current = current[part]
    return current


def normalize_answer_text(text: str) -> str:
    text = text.translate(DIGIT_TRANSLATION)
    text = text.replace("٬", "").replace("،", "").replace(",", "").replace("_", "")
    text = re.sub(r"(?<=\d)[ ](?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", text)
    return text


def has_number(text: str, target) -> bool:
    wanted = parse_number(target)
    if wanted is None:
        return False
    for match in re.findall(r"-?\d+(?:\.\d+)?", normalize_answer_text(text)):
        # Transaction rows are signed. "45000" and "-45000" are the same reported amount.
        if abs(abs(float(match)) - abs(wanted)) < 1e-6:
            return True
    return False


def regex_holds(text: str, pattern: str, mode: str) -> bool:
    if not pattern:
        return True
    found = re.search(pattern, text, flags=re.IGNORECASE) is not None
    if mode == "any":
        return found
    if mode == "none":
        return not found
    raise ValueError(mode)


def judge(task: dict, world: dict, calls: list[dict], answer: str) -> dict:
    incorrect = sum(1 for call in calls if call["kind"] != "ok")
    forbidden = sum(1 for call in calls if call["kind"] == "forbidden")
    blob = "\n".join(json.dumps(call["result"], ensure_ascii=False, sort_keys=True) for call in calls)
    reasons = []
    if forbidden:
        reasons.append("forbidden_call")
    for target in task.get("numbers_all") or []:
        if not has_number(answer, target):
            reasons.append(f"missing_number:{target}")
    if not regex_holds(answer, task.get("answer_regex_any") or "", "any"):
        reasons.append("missing_outcome_phrase")
    if not regex_holds(answer, task.get("answer_regex_none") or "", "none"):
        reasons.append("forbidden_phrase")
    for snippet in task.get("ground_all") or []:
        if snippet not in blob:
            reasons.append(f"ungrounded:{snippet}")
    ground_any = task.get("ground_any") or []
    if ground_any and not any(snippet in blob for snippet in ground_any):
        reasons.append("ungrounded_any")
    for path, expected in (task.get("state") or {}).items():
        actual = lookup_path(world, path)
        if not values_equal(expected, actual):
            reasons.append(f"state:{path}")
    return {
        "success": not reasons,
        "incorrect_tool_calls": incorrect,
        "forbidden_tool_calls": forbidden,
        "reasons": reasons,
    }


def apply_call(world: dict, task: dict, counts: dict, name: str, args) -> dict:
    kind = classify_call(task, counts, name, args if isinstance(args, dict) else None)
    if isinstance(args, dict) and name in {
        "get_balance",
        "list_recent_transactions",
        "transfer",
        "get_card_status",
        "read_sensor",
        "get_alarm_limits",
        "set_alarm_limit",
    }:
        result = dispatch(world, name, args)
    else:
        result = tool_error("invalid_arguments" if name else "unknown_tool", tool=name)
        if kind == "ok":
            kind = "incorrect"
    return {"name": name, "arguments": args, "kind": kind, "result": result}


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel\|>thought.*?<\|channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_arguments(raw):
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def extract_tool_calls(message: dict) -> list[dict]:
    extracted = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        extracted.append(
            {
                "id": call.get("id") or f"call_{index}",
                "name": function.get("name") or "",
                "arguments": parse_arguments(function.get("arguments")),
            }
        )
    return extracted


def post_json(base_url: str, path: str, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from exc


def completion_tokens(payload: dict, fallback_text: str, base_url: str) -> tuple[int, int, str]:
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens_n = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens_n, int):
        return prompt_tokens, completion_tokens_n, "usage"
    token_count = tokenize_count(base_url, fallback_text)
    if token_count is None:
        return 0, 0, "missing"
    return 0, token_count, "tokenize_completion_only"


def tokenize_count(base_url: str, text: str) -> int | None:
    for key in ("content", "prompt"):
        try:
            payload = post_json(base_url, "/tokenize", {key: text}, timeout=60)
        except Exception:
            continue
        tokens = payload.get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    return None


def chat_round(base_url: str, spec: dict, messages: list[dict]) -> dict:
    protocol = spec["protocol"]
    body = {
        "messages": messages,
        "tools": spec["tools"],
        "tool_choice": protocol["tool_choice"],
        "temperature": protocol["temperature"],
        "top_p": protocol["top_p"],
        "max_tokens": protocol["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": protocol["enable_thinking"]},
    }
    return post_json(base_url, "/v1/chat/completions", body, timeout=300)


def run_task(base_url: str, spec: dict, task: dict) -> dict:
    world = fresh_world(spec)
    counts: dict = {}
    messages = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": task["user"]},
    ]
    calls = []
    rounds = []
    prompt_tokens = 0
    completion_tokens_n = 0
    token_sources = []
    started = time.perf_counter()
    error = ""
    final_text = ""
    stopped = "max_rounds"
    max_rounds = int(spec["protocol"]["max_rounds"])
    for round_index in range(max_rounds):
        try:
            payload = chat_round(base_url, spec, messages)
        except Exception as exc:  # noqa: BLE001 - one task must not abort the suite
            error = str(exc)
            stopped = "http_error"
            break
        choice = payload["choices"][0]
        message = choice["message"] or {}
        text = message.get("content") or ""
        round_prompt, round_completion, source = completion_tokens(payload, text, base_url)
        prompt_tokens += round_prompt
        completion_tokens_n += round_completion
        token_sources.append(source)
        extracted = extract_tool_calls(message)
        assistant = {"role": "assistant", "content": text}
        if message.get("tool_calls"):
            assistant["tool_calls"] = message["tool_calls"]
        messages.append(assistant)
        round_calls = []
        for call in extracted:
            executed = apply_call(world, task, counts, call["name"], call["arguments"])
            calls.append(executed)
            round_calls.append(executed)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(executed["result"], ensure_ascii=False),
                }
            )
        rounds.append(
            {
                "round": round_index + 1,
                "finish_reason": choice.get("finish_reason"),
                "content": text,
                "calls": round_calls,
                "prompt_tokens": round_prompt,
                "completion_tokens": round_completion,
                "token_source": source,
            }
        )
        if not extracted:
            final_text = strip_think(text)
            stopped = "final"
            break
        final_text = strip_think(text)
    latency = time.perf_counter() - started
    verdict = judge(task, world, calls, final_text)
    return {
        "id": task["id"],
        "domain": task["domain"],
        "success": verdict["success"],
        "incorrect_tool_calls": verdict["incorrect_tool_calls"],
        "forbidden_tool_calls": verdict["forbidden_tool_calls"],
        "reasons": verdict["reasons"],
        "latency_seconds": round(latency, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens_n,
        "tokens_consumed": prompt_tokens + completion_tokens_n,
        "token_sources": token_sources,
        "stopped": stopped,
        "answer": final_text,
        "error": error,
        "calls": calls,
        "rounds": rounds,
    }


def aggregate(task_rows: list[dict]) -> dict:
    successes = sum(1 for row in task_rows if row["success"])
    total = len(task_rows)
    latency = sum(row["latency_seconds"] for row in task_rows)
    tokens = sum(row["tokens_consumed"] for row in task_rows)
    return {
        "n_tasks": total,
        "task_successes": successes,
        "task_success_rate": round(successes / total, 4) if total else 0.0,
        "incorrect_tool_calls": sum(row["incorrect_tool_calls"] for row in task_rows),
        "e2e_latency_seconds": round(latency, 3),
        "e2e_latency_mean_seconds": round(latency / total, 3) if total else 0.0,
        "tokens_consumed": tokens,
        "prompt_tokens": sum(row["prompt_tokens"] for row in task_rows),
        "completion_tokens": sum(row["completion_tokens"] for row in task_rows),
    }


def checkpoint_record(model_name: str) -> dict:
    known = CHECKPOINTS.get(model_name)
    if known is None:
        return {"file_name": "", "quant": "", "bytes": None}
    path = known["path"]
    size = path.stat().st_size if path.exists() else None
    return {"file_name": known["file_name"], "quant": known["quant"], "bytes": size}


def validate_spec(spec: dict) -> None:
    names = [tool["function"]["name"] for tool in spec["tools"]]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate tool name in spec")
    implemented = {
        "get_balance",
        "list_recent_transactions",
        "transfer",
        "get_card_status",
        "read_sensor",
        "get_alarm_limits",
        "set_alarm_limit",
    }
    if set(names) != implemented:
        raise SystemExit(f"tool schema does not match the harness: {sorted(set(names) ^ implemented)}")
    task_ids = [task["id"] for task in spec["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("duplicate task id")
    for task in spec["tasks"]:
        for rule in task["allow"] + task["forbidden"]:
            if rule["tool"] not in implemented:
                raise SystemExit(f"{task['id']} references unknown tool {rule['tool']}")
        re.compile(task.get("answer_regex_any") or "")
        re.compile(task.get("answer_regex_none") or "")


def scripted_task(spec: dict, task_id: str, script: list[tuple[str, dict | None]], answer: str) -> dict:
    task = next(item for item in spec["tasks"] if item["id"] == task_id)
    world = fresh_world(spec)
    counts: dict = {}
    calls = [apply_call(world, task, counts, name, args) for name, args in script]
    verdict = judge(task, world, calls, answer)
    verdict["calls"] = calls
    return verdict


def self_test() -> None:
    spec = load_spec()
    validate_spec(spec)
    assert has_number("balance is 2,500,000 IRR", 2500000)
    assert has_number("موجودی ۲۵۰۰۰۰۰", 2500000)
    assert has_number("2.500.000", 2500000)
    assert not has_number("balance is 250000 IRR", 2500000)
    assert has_number("humidity 71.0", 71)
    assert has_number("amounts are -45000 and 500000", 45000)

    world = fresh_world(spec)
    rejected = dispatch(
        world,
        "set_alarm_limit",
        {"device_id": "SIM-ROOM-1", "channel": "humidity_pct", "low": 30, "high": 90},
    )
    assert rejected["ok"] is False and rejected["applied"] is False
    assert world["devices"]["SIM-ROOM-1"]["alarm_limits"]["humidity_pct"]["high"] == 60.0
    accepted = dispatch(
        fresh_world(spec),
        "set_alarm_limit",
        {"device_id": "SIM-ROOM-1", "channel": "humidity_pct", "low": 30, "high": 65},
    )
    assert accepted["ok"] is True
    overdraft = dispatch(fresh_world(spec), "transfer", {"from_id": "IR1002", "to_id": "IR1001", "amount_irr": 900000})
    assert overdraft["error"] == "insufficient_funds"

    description = next(tool["function"]["description"] for tool in spec["tools"] if tool["function"]["name"] == "set_alarm_limit")
    assert "55" in description and "70" in description
    assert spec["world"]["devices"]["SIM-ROOM-1"]["published_safe_bounds"]["humidity_pct"]["high"] == [55.0, 70.0]

    perfect = {
        "bank_balance": ([("get_balance", {"account_id": "IR1001"})], "The balance is 2500000 IRR."),
        "bank_recent_tx": (
            [("list_recent_transactions", {"account_id": "IR1001", "limit": 2})],
            "The newest amounts are -45000 and 500000.",
        ),
        "bank_card_status": ([("get_card_status", {"card_id": "CARD88"})], "Card CARD88 is blocked."),
        "bank_transfer_if_funds": (
            [
                ("get_balance", {"account_id": "IR1001"}),
                ("transfer", {"from_id": "IR1001", "to_id": "IR1002", "amount_irr": 100000}),
            ],
            "Transferred 100000 IRR. The remaining balance is 2400000 IRR.",
        ),
        "bank_overdraft": (
            [
                ("get_balance", {"account_id": "IR1002"}),
                ("transfer", {"from_id": "IR1002", "to_id": "IR1001", "amount_irr": 900000}),
            ],
            "The transfer was rejected because of insufficient funds.",
        ),
        "bank_unknown_no_transfer": (
            [("get_balance", {"account_id": "IR9999"})],
            "Account IR9999 was not found.",
        ),
        "clinic_humidity_alarm": (
            [
                ("read_sensor", {"device_id": "SIM-ROOM-1", "channel": "humidity_pct"}),
                ("get_alarm_limits", {"device_id": "SIM-ROOM-1"}),
            ],
            "Humidity is 71, which is above the high limit of 60.",
        ),
        "clinic_reject_high_limit": (
            [
                (
                    "set_alarm_limit",
                    {"device_id": "SIM-ROOM-1", "channel": "humidity_pct", "low": 30, "high": 90},
                )
            ],
            "The change was rejected as outside the safe range, so the limit is unchanged.",
        ),
    }
    rows = []
    for task_id, (script, answer) in perfect.items():
        verdict = scripted_task(spec, task_id, script, answer)
        if not verdict["success"] or verdict["incorrect_tool_calls"] != 0:
            raise SystemExit(f"perfect trace failed for {task_id}: {verdict}")
        rows.append({"success": True, "incorrect_tool_calls": 0, "latency_seconds": 1.0, "tokens_consumed": 10, "prompt_tokens": 8, "completion_tokens": 2})

    happened = scripted_task(
        spec,
        "bank_transfer_if_funds",
        [
            ("get_balance", {"account_id": "IR1001"}),
            ("transfer", {"from_id": "IR1001", "to_id": "IR1002", "amount_irr": 100000}),
        ],
        "The transfer happened, and the remaining balance of IR1001 is 2400000.",
    )
    if not happened["success"] or happened["incorrect_tool_calls"] != 0:
        raise SystemExit(f"transfer-happened phrasing should pass: {happened}")

    refused = scripted_task(
        spec,
        "clinic_reject_high_limit",
        [],
        "I will not set 90 because it is outside the safe range. The limit is unchanged.",
    )
    if not refused["success"]:
        raise SystemExit(f"text refusal should succeed: {refused}")

    forbidden = scripted_task(
        spec,
        "bank_unknown_no_transfer",
        [
            ("get_balance", {"account_id": "IR9999"}),
            ("transfer", {"from_id": "IR1001", "to_id": "IR1002", "amount_irr": 100000}),
        ],
        "Account IR9999 was not found.",
    )
    if forbidden["success"] or forbidden["forbidden_tool_calls"] < 1 or forbidden["incorrect_tool_calls"] < 1:
        raise SystemExit(f"forbidden transfer was not counted: {forbidden}")

    wrong_amount = scripted_task(
        spec,
        "bank_overdraft",
        [("transfer", {"from_id": "IR1002", "to_id": "IR1001", "amount_irr": 100000})],
        "The transfer was rejected.",
    )
    if wrong_amount["success"] or "state:accounts.IR1002.balance_irr" not in wrong_amount["reasons"]:
        raise SystemExit(f"smaller transfer should fail the task: {wrong_amount}")

    extra_read = scripted_task(
        spec,
        "bank_balance",
        [
            ("get_card_status", {"card_id": "CARD88"}),
            ("get_balance", {"account_id": "IR1001"}),
        ],
        "The balance is 2,500,000 IRR.",
    )
    if not extra_read["success"] or extra_read["incorrect_tool_calls"] != 1:
        raise SystemExit(f"extra card read should be incorrect but not fatal: {extra_read}")

    summary = aggregate(rows)
    if summary["task_success_rate"] != 1.0 or summary["incorrect_tool_calls"] != 0:
        raise SystemExit(summary)
    print("self-test ok", flush=True)


def rejudge_report(path: Path) -> None:
    """Apply the current answer rules to a saved trace without calling the model."""
    spec = load_spec()
    validate_spec(spec)
    report = json.loads(path.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in spec["tasks"]}
    for row in report["tasks"]:
        task = tasks[row["id"]]
        world = fresh_world(spec)
        counts: dict = {}
        calls = [apply_call(world, task, counts, call["name"], call["arguments"]) for call in row["calls"]]
        verdict = judge(task, world, calls, row["answer"])
        row["calls"] = calls
        row["success"] = verdict["success"]
        row["incorrect_tool_calls"] = verdict["incorrect_tool_calls"]
        row["forbidden_tool_calls"] = verdict["forbidden_tool_calls"]
        row["reasons"] = verdict["reasons"]
    report.update(aggregate(report["tasks"]))
    report["complete"] = len(report["tasks"]) == len(spec["tasks"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate(report["tasks"]), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--server-flags", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rejudge", default="")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rejudge:
        rejudge_report(Path(args.rejudge))
        return
    if not args.model_name:
        raise SystemExit("--model-name is required")

    spec = load_spec()
    validate_spec(spec)
    tasks = spec["tasks"]
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [task for task in tasks if task["id"] in wanted]
        if not tasks:
            raise SystemExit(f"no tasks matched {args.only}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS / f"tool-{args.model_name}.json"
    report = {
        "model": args.model_name,
        "base_url": args.base_url,
        "checkpoint": checkpoint_record(args.model_name),
        "server_flags": args.server_flags,
        "protocol": spec["protocol"],
        "spec": "data/tool_tasks.json",
        "quants_matched": False,
        "tasks": [],
    }
    for index, task in enumerate(tasks, start=1):
        row = run_task(args.base_url, spec, task)
        report["tasks"].append(row)
        report.update(aggregate(report["tasks"]))
        report["complete"] = len(report["tasks"]) == len(spec["tasks"])
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{args.model_name} {task['id']} {index}/{len(tasks)} "
            f"success={int(row['success'])} incorrect={row['incorrect_tool_calls']} "
            f"{row['latency_seconds']}s tokens={row['tokens_consumed']} reasons={row['reasons']}",
            flush=True,
        )
    print(json.dumps(aggregate(report["tasks"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
