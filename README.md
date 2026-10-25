# Nemotron 3.5 Bolt E4B vs Gemma 4 E4B and E2B on Persian

Short result: on this machine, **Nemotron 3.5 Bolt E4B (Unsloth UD-Q4_K_XL)** is the stronger Persian model. **Gemma 4 E4B (Q5_K_M)** is the same size class and finishes second. **Gemma 4 E2B (Q5_K_S)** is the one that fits a 4 GB GPU more comfortably. Across five frozen Persian probes the macro means are **0.781 vs 0.734 vs 0.638**.

| Task | Metric | Nemotron UD-Q4_K_XL | Gemma 4 E4B Q5_K_M | Gemma 4 E2B Q5_K_S |
|---|---|---:|---:|---:|
| Belebele Persian reading comprehension | accuracy | **1.000** | **1.000** | 0.625 |
| ParsiNLU food aspect sentiment | accuracy | 0.500 | 0.500 | 0.500 |
| ParsiNLU textual entailment | accuracy | **0.750** | **0.750** | 0.500 |
| ParsiNLU query paraphrase | accuracy | 0.750 | 0.625 | **0.875** |
| PersianQA extractive QA | token F1 | **0.906** | 0.796 | 0.691 |
| Macro mean of the five scores | — | **0.781** | 0.734 | 0.638 |

Use Nemotron when Persian reading, entailment, or answer extraction matters. Gemma 4 E4B matches Nemotron on reading and entailment and sits between the other two on QA. Use Gemma 4 E2B when the job is mostly “are these two questions the same?”.

Weights are not in this repository. The scorer, the frozen 40-item sample, and the raw predictions are.

## Models

All three are Gemma 4 text models served with llama.cpp, Jinja chat templates, thinking disabled, temperature 0, and top-p 0.95.

| | Nemotron 3.5 Bolt E4B | Gemma 4 E4B | Gemma 4 E2B |
|---|---|---|---|
| Lineage | NVIDIA post-train of Gemma 4 E4B-it | Google Gemma 4 E4B-it | Google Gemma 4 E2B-it |
| Effective size | about 4B active, about 8B stored | about 4B active, about 8B stored | about 2B effective, about 5B stored |
| File | `NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf` | `gemma-4-E4B-it-Q5_K_M.gguf` | `gemma-4-E2B-it-Q5_K_S.gguf` |
| Source | `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` | local Gemma 4 E4B GGUF | local Unsloth Q5_K_S build |
| Bytes on disk | 5,060,482,240 | 5,481,796,768 | about 3.09 GB |
| Quantization | Unsloth Dynamic Q4_K_XL | Q5_K_M | Q5_K_S |
| Context used | 1536 | 1536 | 1536 |
| GPU layers requested | 99 | 99 | 99 |

The quantizations are not matched. Gemma 4 E4B is the base model Nemotron was post-trained from, stored at higher bits per weight (Q5_K_M versus UD-Q4_K_XL). A Nemotron win over that file is therefore not a gift of a more generous quant. Gemma 4 E2B is a smaller network, also stored at Q5. A win for E2B is not a pure architecture win either.

Nemotron and Gemma 4 E4B did not fit in 4 GB. llama.cpp reported that a forced 99-layer offload could not meet the free-memory target, then still mapped what it could onto the GPU. Observed use for each was about **3.8 GB of 4 GB** VRAM, with the rest of the weights CPU-mapped. Steady generation after the first compiled prompt was about **21 tokens/s** for Nemotron. Gemma 4 E2B used about **2.2 GB** VRAM and, after warmup, about **35–39 tokens/s**.

The Nemotron numbers, the Gemma 4 E4B numbers, and the writing probes for those two models used llama.cpp `b11871` (CUDA 12.4 Windows build). The Gemma 4 E2B classification numbers used an earlier CUDA build, `b9102`, on the same frozen prompts, the same sampling, and the same context. The E2B writing probe was repeated on `b11871`. Treat the E2B classification gap as real on this sample, and treat that runtime mismatch as a disclosed confound, not as a hidden one.

## Hardware

- GPU: NVIDIA GeForce GTX 1650, 4096 MiB, compute capability 7.5, no tensor cores
- Host: Windows, 16 GB RAM
- One model resident at a time

The 5.06 GB and 5.48 GB checkpoints cannot stay fully resident, so part of the weights stay CPU-mapped. The E2B checkpoint can stay much closer to fully on-device.

## Protocol

`eval_fa.py` draws one shared sample and scores every model on it.

- Seed 25, 8 items per dataset, 40 items total
- Labels are balanced across the labels present in the fetched slice, except PersianQA, which is a seeded shuffle of rows that actually contain an answer string
- Greedy decoding: temperature 0, top-p 0.95
- `enable_thinking` false, and the servers were also started with `--reasoning off`
- Classification max tokens 16. QA max tokens 48
- Accuracy is exact label match after a small parser (letter, sentiment code, entailment letter, or 0/1)
- PersianQA score is token F1 against any gold string, after Arabic yeh/kaf normalization

This is a controlled probe. It is not a full test-set leaderboard. With 8 items, a swing of one item moves accuracy by 0.125. The Nemotron-versus-E2B order repeats an earlier pass on the official NVIDIA `Q4_K_M` file: Nemotron ahead on reading, entailment, and QA, E2B ahead only on paraphrase, sentiment tied. Gemma 4 E4B was scored once, on this same frozen sample.

## Datasets

The frozen prompts are in `data/samples.json`. They were taken from public Hugging Face splits through the dataset viewer, then held fixed.

| Key | Upstream | Split | Task and score |
|---|---|---|---|
| `belebele_fa` | `facebook/belebele`, config `pes_Arab` | test | 4-way multiple choice. Gold is `A`–`D` from `correct_answer_num`. |
| `parsinlu_sentiment_food` | `persiannlp/parsinlu_sentiment` | `test_food` | Aspect sentiment. Gold codes observed here include `-3,-2,-1,0,1,2,3`. `-3` is the “aspect not stated” bin used by this probe. |
| `parsinlu_entailment` | `persiannlp/parsinlu_entailment` | test | `e` entailment, `c` contradiction, `n` neutral. |
| `parsinlu_paraphrase` | `persiannlp/parsinlu_query_paraphrasing` | test | `1` paraphrase, `0` not. |
| `persian_qa` | `SajjadAyoubi/persian_qa` | validation | Extractive QA. Context is windowed around `answer_start` so the gold span is inside the prompt. Empty answer rows were dropped. |

Rows with an empty PersianQA answer were common in the viewer slice (60 of 200). Scoring those as F1 against an empty gold would punish any answer, so they are excluded. The published sample only contains rows with at least one non-empty gold string.

## Error analysis

Sentiment is a three-way tie at 0.50, and it is a bad tie. Nemotron’s four misses were gold `-2`, `-3`, `0`, and `3`, all predicted as `-1`. Gemma 4 E4B missed the same four golds, three as `-1` and the `-3` as `-2`. The Persian wording of the review is not the bottleneck. The label inventory is.

Entailment errors are directional. Nemotron and Gemma 4 E4B each missed the same way: a contradiction and a neutral, both predicted as entailment (`e`). Both finished at 0.75 against E2B’s 0.50.

Paraphrase is E2B’s only win, 0.875 vs Nemotron’s 0.750 vs E4B’s 0.625. The larger models’ misses were false negatives: gold `1`, prediction `0` (two for Nemotron, three for E4B). E2B is the better “same question?” detector on this slice.

PersianQA is the clearest quality gap. Nemotron’s token F1 is 0.906, with seven exact or full-credit answers and one partial (`سرپرستی ابراهیم` against a gold span that names Terah taking Abraham into care, F1 0.25). Gemma 4 E4B is 0.796. Its partials add “دوک اتریش” to Leopold V (F1 0.33), expand “مشخص نیست” (F1 0.67), and again miss the Terah/Abraham care span, answering that Abraham took charge (F1 0.36). E2B’s F1 is 0.691.

Belebele was perfect for Nemotron and for Gemma 4 E4B (8/8, letters only) and 5/8 for E2B. One E2B miss on this slice was gold `A` predicted as `C`. Do not read 1.000 as a claim about the full 900-row Belebele Persian test.

## Persian generation probe

Five shared instructions, temperature 0, 120 token cap, thinking off. Full text is in `results/qual-nemotron.json`, `results/qual-gemma-e4b.json`, and `results/qual-gemma.json`.

| Probe | Nemotron | Gemma 4 E4B | Gemma 4 E2B |
|---|---|---|---|
| Formal rewrite of a colloquial sentence | Keeps “عصبی” and produces one clean formal sentence. | Keeps “عصبی” and produces one clean formal sentence. | Rewrites “عصبی” as “مضطرب”, which changes angry into anxious. |
| Grammar: گرفتن → گرفتند | Correct, only the sentence. | Fixes the verb, but drops «به» before «امانت». | Correct, only the sentence. |
| Translate “must not invent dates” | Wrong lexical choice: “نباید تاریخ اختراع کند”. | Same wrong choice: “نباید تاریخ اختراع کند”. | Still a calque, but number and object marking survive: “نباید تاریخ‌ها را اختراع کند”. |
| Proverb «هر که بامش بیش برفش بیشتر» | Right idea: more wealth or rank, more responsibility and trouble. | Right idea: higher rank, more responsibility and trouble. | Wrong idea: more capacity, more consumption. |
| Two Persian sentences about a 10:00 meeting, no English | Two sentences, no extra English, no added “صبح”. | Two sentences, no extra English, no added “صبح”. | Fluent and more formal, but adds a preamble and adds “صبح”, which the instruction did not say. |

Nemotron and Gemma 4 E4B follow the short Persian instructions and know the proverb. E4B drops one preposition in the grammar fix and makes the same “تاریخ اختراع” calque as Nemotron. E2B is the less-wrong translator of “invent dates” and the one that misses the proverb. None of the translations is something to ship without a glossary.

## Decision

For a Persian assistant on this hardware, run Nemotron 3.5 Bolt E4B UD-Q4_K_XL if answer correctness is the product. Budget for hybrid execution, because the checkpoint does not stay fully on the 1650. Gemma 4 E4B Q5_K_M is the same size class, stored at higher bits, and it does not overtake that post-train on this sample.

Run Gemma 4 E2B if the 1650 is the production GPU and paraphrase detection is the job. It wins that task. It is not the better reader or QA model here.

Do not use any of these raw checkpoints as an aspect-sentiment classifier. That task needs a constrained label head or a verifier. The 0.50 tie is the evidence.

## Reproduce

Python 3.10+ standard library only. Point the client at an OpenAI-compatible llama.cpp server that has already loaded the checkpoint you want to score.

```bash
python eval_fa.py --base-url http://127.0.0.1:8020 --model-name nemotron-ud-q4-k-xl
python eval_fa.py --base-url http://127.0.0.1:8030 --model-name gemma-4-e4b
python eval_fa.py --base-url http://127.0.0.1:8020 --model-name gemma-4-e2b
```

`--prepare-only` rebuilds `data/samples.json` from the dataset viewer. Do not do that if you want to compare against the numbers in this file. The published sample is the measurement.

Server flags used for the Nemotron run:

```text
llama serve -m NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8020 -c 1536 -ngl 99 --reasoning off
```

Equivalent Hub selector, if the file is not already local:

```text
llama serve -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL
```

## Layout

- `eval_fa.py` — fetch, freeze, score
- `data/samples.json` — the 40 prompts and gold labels
- `results/nemotron-ud-q4-k-xl.json` — per-item Nemotron predictions
- `results/gemma-4-e4b.json` — per-item Gemma 4 E4B predictions
- `results/gemma-4-e2b.json` — per-item Gemma 4 E2B predictions
- `results/qual-nemotron.json`, `results/qual-gemma-e4b.json`, `results/qual-gemma.json` — the five writing probes
- `eval_tools.py` — simulated tool-calling harness
- `data/tool_tasks.json` — frozen world, tool schemas, eight tasks, and success rules
- `results/tool-nemotron-ud-q4-k-xl.json`, `results/tool-gemma-4-e4b.json` — tool-calling traces

## Tool-calling comparison

Same llama.cpp `b11871` server, Jinja chat template, thinking off, temperature 0, top-p 0.95, context 1536. One model resident at a time. Each scored run started from a freshly loaded server. The agent loop stops after 4 model rounds. Tools are in-process fakes with one seeded world per task: a simulated bank, and a simulated clinic storeroom monitor. Nothing here calls a real bank or a real device, and the alarm setter rejects limits outside a published safe range.

The quantizations are not matched. Nemotron is Unsloth UD-Q4_K_XL (5,060,482,240 bytes). Gemma 4 E4B is Q5_K_M (5,481,796,768 bytes), so Gemma is stored at higher bits per weight.

| Metric | Nemotron UD-Q4_K_XL | Gemma 4 E4B Q5_K_M |
|---|---:|---:|
| Task success rate | 8/8 | 8/8 |
| Incorrect tool calls | 0 | 0 |
| End-to-end latency | 71.747 s | 75.413 s |
| Tokens consumed | 14,610 | 14,681 |

Latency is the sum of the eight task wall times, including every tool round. The mean is 8.968 s for Nemotron and 9.427 s for Gemma. Tokens are the sum of `prompt_tokens` and `completion_tokens` from the llama.cpp usage field on every completion. The tool-schema prefix is counted again on each round, which is why a short answer still costs about 1,600 tokens. No tokenize fallback was needed.

The first task in each run compiled that shared prefix and took about 25 s. Later tasks reused the prefix cache. The suite total is not eight cold starts.

No task failed for either model. Both used the right tool and the right arguments on the balance, the two newest transactions, the blocked card, the unknown account, and the humidity alarm. Both checked IR1001 and then transferred exactly 100000 IRR, leaving 2400000. On the out-of-range humidity limit, both read the published bounds and refused in text. Neither called `set_alarm_limit`, and the stored high limit stayed 60.

The only behavioral difference is the overdraft. Nemotron read IR1002 (400000 IRR) and did not call `transfer`. Gemma called `transfer` for the requested 900000 IRR, the fake tool returned `insufficient_funds`, and Gemma reported that. Under the frozen rules that call is allowed, because the amount and accounts match the request and the tool rejects it. Neither model moved a smaller unauthorized amount, and neither transferred after being told not to.

This suite is saturated. It does not separate the models on success or on incorrect calls. The latency and token gaps are small, and they are not a pure architecture result, because the quants differ and part of each 5 GB checkpoint stays CPU-mapped.

### Domain GRPO was not run

These figures are the pre-RL baseline. Domain-specific RL with NeMo Gym and GRPO was not executed, and there are no post-RL scores. This GTX 1650 cannot hold either 5 GB checkpoint fully for inference: a forced 99-layer offload misses the free-memory target, and observed use is about 3.8 GB of 4 GB VRAM. A GRPO job on that GPU would be a false run.

The proposed next experiment is NeMo Gym GRPO on simulated domains of this kind, for example a banking customer-service agent or this storeroom-monitor console. The eight tasks above cannot show a training gain, because both models already succeed on all of them with no incorrect call. A harder held-out set would have to come first.

```bash
python eval_tools.py --self-test
python eval_tools.py --base-url http://127.0.0.1:8020 --model-name nemotron-ud-q4-k-xl
python eval_tools.py --base-url http://127.0.0.1:8030 --model-name gemma-4-e4b
```

Serve one checkpoint at a time with the same flags as the Persian runs (`-c 1536 -ngl 99 --reasoning off`). Do not rebuild `data/tool_tasks.json` if you want to compare against this table.
