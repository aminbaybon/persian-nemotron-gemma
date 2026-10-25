"""Compare a local llama-server model on five Persian datasets.

Each dataset uses the same 8-item sample (seed 25) for every model.
Classification tasks are scored by exact label match. PersianQA uses
token-level F1 against the gold answer strings.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "samples.json"
RESULTS = ROOT / "results"

N_PER_DATASET = 8
SEED = 25
VIEWER = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{VIEWER}?{query}", timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all(dataset: str, config: str, split: str, cap: int = 400) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while offset < cap:
        page = fetch_rows(dataset, config, split, offset, 100)
        batch = [item["row"] for item in page["rows"]]
        rows.extend(batch)
        total = page.get("num_rows_total") or 0
        offset += len(batch)
        if not batch or offset >= total or offset >= cap:
            break
    return rows


def balanced(rows: list[dict], key_fn, n: int, rng: random.Random) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row))].append(row)
    for group in buckets.values():
        rng.shuffle(group)
    picked: list[dict] = []
    while len(picked) < n and any(buckets.values()):
        for label in sorted(buckets):
            if buckets[label] and len(picked) < n:
                picked.append(buckets[label].pop())
    return picked


def build_samples() -> dict:
    rng = random.Random(SEED)
    bele = fetch_all("facebook/belebele", "pes_Arab", "test", cap=200)
    food = fetch_all("persiannlp/parsinlu_sentiment", "parsinlu-repo", "test_food", cap=300)
    entail = fetch_all("persiannlp/parsinlu_entailment", "parsinlu-repo", "test", cap=300)
    para = fetch_all("persiannlp/parsinlu_query_paraphrasing", "parsinlu-repo", "test", cap=300)
    qa = fetch_all("SajjadAyoubi/persian_qa", "persian_qa", "validation", cap=200)

    samples = {
        "belebele_fa": [],
        "parsinlu_sentiment_food": [],
        "parsinlu_entailment": [],
        "parsinlu_paraphrase": [],
        "persian_qa": [],
    }
    for row in balanced(bele, lambda r: r["correct_answer_num"], N_PER_DATASET, rng):
        samples["belebele_fa"].append(
            {
                "prompt": (
                    "متن زیر را بخوان و فقط حرف گزینه درست را بنویس (A یا B یا C یا D).\n\n"
                    f"متن: {row['flores_passage']}\n"
                    f"پرسش: {row['question']}\n"
                    f"A) {row['mc_answer1']}\n"
                    f"B) {row['mc_answer2']}\n"
                    f"C) {row['mc_answer3']}\n"
                    f"D) {row['mc_answer4']}\n"
                    "پاسخ:"
                ),
                "gold": "ABCD"[int(row["correct_answer_num"]) - 1],
                "task": "choice",
            }
        )
    for row in balanced(food, lambda r: r["label"], N_PER_DATASET, rng):
        samples["parsinlu_sentiment_food"].append(
            {
                "prompt": (
                    "احساس نظر نسبت به جنبه خواسته‌شده را فقط با یکی از این کدها بنویس: "
                    "-3 (جنبه در متن نیست)، -2 (خیلی منفی)، -1 (منفی)، 0 (خنثی)، "
                    "1 (مثبت)، 2 (خیلی مثبت)، 3 (بسیار مثبت).\n\n"
                    f"نظر: {row['review']}\n"
                    f"پرسش: {row['question']}\n"
                    f"جنبه: {row.get('aspect') or ''}\n"
                    "کد:"
                ),
                "gold": str(row["label"]),
                "task": "sentiment",
            }
        )
    label_fa = {"e": "استلزام (e)", "c": "تناقض (c)", "n": "خنثی (n)"}
    for row in balanced(entail, lambda r: r["label"], N_PER_DATASET, rng):
        samples["parsinlu_entailment"].append(
            {
                "prompt": (
                    "رابطه جمله دوم با جمله اول را فقط با یک حرف بنویس: "
                    "e برای استلزام، c برای تناقض، n برای خنثی.\n\n"
                    f"جمله ۱: {row['sent1']}\n"
                    f"جمله ۲: {row['sent2']}\n"
                    "برچسب:"
                ),
                "gold": str(row["label"]).lower(),
                "task": "entail",
                "gold_name": label_fa.get(str(row["label"]).lower(), str(row["label"])),
            }
        )
    for row in balanced(para, lambda r: r["label"], N_PER_DATASET, rng):
        samples["parsinlu_paraphrase"].append(
            {
                "prompt": (
                    "آیا این دو پرسش هم‌معنی هستند؟ فقط 1 (بله) یا 0 (خیر) بنویس.\n\n"
                    f"پرسش ۱: {row['q1']}\n"
                    f"پرسش ۲: {row['q2']}\n"
                    "برچسب:"
                ),
                "gold": str(row["label"]),
                "task": "binary",
            }
        )
    rng.shuffle(qa)
    answered = []
    for row in qa:
        answers = row.get("answers") or {}
        texts = answers.get("text") if isinstance(answers, dict) else None
        if isinstance(texts, str):
            texts = [texts]
        texts = [text for text in (texts or []) if str(text).strip()]
        if texts:
            row = dict(row)
            row["_answers"] = list(dict.fromkeys(texts))
            answered.append(row)
    for row in answered[:N_PER_DATASET]:
        context = row["context"]
        if len(context) > 900:
            context = context[:900]
        samples["persian_qa"].append(
            {
                "prompt": (
                    "فقط با عبارت کوتاه داخل متن به پرسش جواب بده.\n\n"
                    f"متن: {context}\n"
                    f"پرسش: {row['question']}\n"
                    "پاسخ:"
                ),
                "gold": row["_answers"],
                "task": "qa",
            }
        )
    return samples


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel\|>thought.*?<\|channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


def chat(base_url: str, prompt: str, max_tokens: int) -> str:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return strip_think(payload["choices"][0]["message"]["content"] or "")


def norm(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    text = re.sub(r"[^\w\u0600-\u06FF+-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [part for part in norm(text).split(" ") if part]


def f1(pred: str, golds: list[str]) -> float:
    pred_tokens = tokens(pred)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in golds:
        gold_tokens = tokens(gold)
        if not gold_tokens:
            continue
        common = 0
        pool = gold_tokens.copy()
        for token in pred_tokens:
            if token in pool:
                common += 1
                pool.remove(token)
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score_example(task: str, pred: str, gold) -> float:
    raw = pred.strip()
    if task == "choice":
        match = re.search(r"[ABCD]", raw.upper())
        return 1.0 if match and match.group(0) == gold else 0.0
    if task == "sentiment":
        match = re.search(r"-?[0-3]", raw)
        return 1.0 if match and match.group(0) == str(gold) else 0.0
    if task == "entail":
        lowered = raw.lower()
        if "تناقض" in raw:
            found = "c"
        elif "استلزام" in raw:
            found = "e"
        elif "خنثی" in raw or "خنثي" in raw:
            found = "n"
        else:
            match = re.search(r"[ecn]", lowered)
            found = match.group(0) if match else ""
        return 1.0 if found == gold else 0.0
    if task == "binary":
        if re.search(r"خیر|نيست|نیست", raw) and not re.search(r"\b1\b", raw):
            found = "0"
        elif re.search(r"بله", raw) and not re.search(r"\b0\b", raw):
            found = "1"
        else:
            match = re.search(r"[01]", raw)
            found = match.group(0) if match else ""
        return 1.0 if found == str(gold) else 0.0
    if task == "qa":
        return f1(raw, gold)
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model-name", default="prepare")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated dataset keys to run")
    args = parser.parse_args()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not args.prepare_only:
        samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    else:
        samples = build_samples()
        DATA_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"wrote {DATA_PATH}")
        return

    if args.only:
        samples = {key: samples[key] for key in args.only.split(",") if key}
    out_path = RESULTS / f"{args.model_name}.json"
    if args.only and out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        report = {"model": args.model_name, "base_url": args.base_url, "datasets": {}}
    report["base_url"] = args.base_url
    for name, items in samples.items():
        rows = []
        started = time.time()
        for index, item in enumerate(items, start=1):
            max_tokens = 48 if item["task"] == "qa" else 16
            t0 = time.time()
            try:
                pred = chat(args.base_url, item["prompt"], max_tokens)
                value = score_example(item["task"], pred, item["gold"])
                error = ""
            except Exception as exc:  # noqa: BLE001 - keep the rest of the run
                pred = ""
                value = 0.0
                error = str(exc)
            rows.append(
                {
                    "i": index,
                    "gold": item["gold"],
                    "pred": pred,
                    "score": value,
                    "seconds": round(time.time() - t0, 2),
                    "error": error,
                }
            )
            print(f"{args.model_name} {name} {index}/{len(items)} score={value:.2f} {rows[-1]['seconds']}s", flush=True)
        metric = sum(row["score"] for row in rows) / len(rows)
        report["datasets"][name] = {
            "n": len(rows),
            "metric": round(metric, 4),
            "metric_name": "token_f1" if items[0]["task"] == "qa" else "accuracy",
            "seconds": round(time.time() - started, 1),
            "rows": rows,
        }
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {metric:.3f}", flush=True)
    print(json.dumps({k: v["metric"] for k, v in report["datasets"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
