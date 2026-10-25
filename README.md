# Nemotron 3.5 Bolt E4B vs Gemma 4 E2B on Persian

Short result: on this machine, **Nemotron 3.5 Bolt E4B (Unsloth UD-Q4_K_XL)** is the stronger Persian model. **Gemma 4 E2B (Q5_K_S)** is the faster one and the one that fits a 4 GB GPU more comfortably. Across five frozen Persian probes the macro mean is **0.781 vs 0.638**.

| Task | Metric | Nemotron UD-Q4_K_XL | Gemma 4 E2B Q5_K_S |
|---|---|---:|---:|
| Belebele Persian reading comprehension | accuracy | **1.000** | 0.625 |
| ParsiNLU food aspect sentiment | accuracy | 0.500 | 0.500 |
| ParsiNLU textual entailment | accuracy | **0.750** | 0.500 |
| ParsiNLU query paraphrase | accuracy | 0.750 | **0.875** |
| PersianQA extractive QA | token F1 | **0.906** | 0.691 |
| Macro mean of the five scores | — | **0.781** | 0.638 |
| Wall time, 40 requests | seconds | 205.7 | 28.6 |

Use Nemotron when Persian reading, entailment, or answer extraction matters. Use Gemma 4 E2B when latency on a 4 GB laptop GPU matters, or when the job is mostly “are these two questions the same?”.

Weights are not in this repository. The scorer, the frozen 40-item sample, and the raw predictions are.

## Models

Both models are Gemma 4 text models served with llama.cpp, Jinja chat templates, thinking disabled, temperature 0, and top-p 0.95.

| | Nemotron 3.5 Bolt E4B | Gemma 4 E2B |
|---|---|---|
| Lineage | NVIDIA post-train of Gemma 4 E4B-it | Google Gemma 4 E2B-it |
| Effective size | about 4B active, about 8B stored | about 2B effective, about 5B stored |
| File | `NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf` | `gemma-4-E2B-it-Q5_K_S.gguf` |
| Source | `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` | local Unsloth Q5_K_S build |
| Bytes on disk | 5,060,482,240 | about 3.09 GB |
| Quantization | Unsloth Dynamic Q4_K_XL | Q5_K_S |
| Context used | 1536 | 1536 |
| GPU layers requested | 99 | 99 |

The quantizations are not matched. Gemma is stored at higher bits per weight. Nemotron is a larger network stored more aggressively. A win for Nemotron is therefore not a pure architecture win, and a win for Gemma is not a pure quantization win.

Nemotron did not fit in 4 GB. llama.cpp reported that a forced 99-layer offload could not meet the free-memory target, then still mapped what it could onto the GPU. Observed use was about **3.8 GB of 4 GB** VRAM, with the rest of the weights CPU-mapped. Steady generation after the first compiled prompt was about **21 tokens/s**. Gemma used about **2.2 GB** VRAM and, after warmup, about **35–39 tokens/s**.

The Nemotron numbers and the Persian writing probe used llama.cpp `b11871` (CUDA 12.4 Windows build). The Gemma classification numbers used an earlier CUDA build, `b9102`, on the same frozen prompts, the same sampling, and the same context. The Gemma writing probe was repeated on `b11871`. Treat the classification gap as real on this sample, and treat the runtime mismatch as a disclosed confound, not as a hidden one.

## Hardware

- GPU: NVIDIA GeForce GTX 1650, 4096 MiB, compute capability 7.5, no tensor cores
- Host: Windows, 16 GB RAM
- One model resident at a time

The 1650 is the reason the speed gap is large. A 5.06 GB checkpoint cannot stay fully resident, so prompt processing pays for CPU/GPU splits. Gemma’s checkpoint can stay much closer to fully on-device.

## Protocol

`eval_fa.py` draws one shared sample and scores every model on it.

- Seed 25, 8 items per dataset, 40 items total
- Labels are balanced across the labels present in the fetched slice, except PersianQA, which is a seeded shuffle of rows that actually contain an answer string
- Greedy decoding: temperature 0, top-p 0.95
- `enable_thinking` false, and the Nemotron server was also started with `--reasoning off`
- Classification max tokens 16. QA max tokens 48
- Accuracy is exact label match after a small parser (letter, sentiment code, entailment letter, or 0/1)
- PersianQA score is token F1 against any gold string, after Arabic yeh/kaf normalization

This is a controlled probe. It is not a full test-set leaderboard. With 8 items, a swing of one item moves accuracy by 0.125. The ranking is still informative because it repeats an earlier pass on the official NVIDIA `Q4_K_M` file: same order, Nemotron ahead on reading, entailment, and QA, Gemma ahead only on paraphrase, sentiment tied.

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

Sentiment is a tie at 0.50, and it is a bad tie. Both models collapse toward `-1` on the fine scale. Nemotron’s four misses were gold `-2`, `-3`, `0`, and `3`, all predicted as `-1`. The Persian wording of the review is not the bottleneck. The label inventory is.

Entailment errors are directional. Nemotron’s two misses were a contradiction and a neutral, both predicted as entailment (`e`). It still finished at 0.75 against Gemma’s 0.50.

Paraphrase is Gemma’s only win, 0.875 vs 0.750. Nemotron’s two misses were false negatives: gold `1`, prediction `0`. Gemma is the better “same question?” detector on this slice.

PersianQA is the clearest quality gap. Nemotron’s token F1 is 0.906, with seven exact or full-credit answers and one partial (`سرپرستی ابراهیم` against a gold span that names Terah taking Abraham into care, F1 0.25). Gemma’s F1 is 0.691. On a 4 GB card this is also the expensive task for Nemotron: 76 seconds for eight questions versus 10 seconds for Gemma.

Belebele was perfect for Nemotron (8/8, letters only, no extra prose) and 5/8 for Gemma. One Gemma miss on this slice was gold `A` predicted as `C`. Do not read 1.000 as a claim about the full 900-row Belebele Persian test.

## Persian generation probe

Five shared instructions, temperature 0, 120 token cap, thinking off. Full text is in `results/qual-nemotron.json` and `results/qual-gemma.json`.

| Probe | Nemotron | Gemma 4 E2B |
|---|---|---|
| Formal rewrite of a colloquial sentence | Keeps “عصبی” and produces one clean formal sentence. | Rewrites “عصبی” as “مضطرب”, which changes angry into anxious. |
| Grammar: گرفتن → گرفتند | Correct, only the sentence. | Correct, only the sentence. |
| Translate “must not invent dates” | Wrong lexical choice: “نباید تاریخ اختراع کند”. | Still a calque, but number and object marking survive: “نباید تاریخ‌ها را اختراع کند”. |
| Proverb «هر که بامش بیش برفش بیشتر» | Right idea: more wealth or rank, more responsibility and trouble. | Wrong idea: more capacity, more consumption. |
| Two Persian sentences about a 10:00 meeting, no English | Two sentences, no extra English, no added “صبح”. | Fluent and more formal, but adds a preamble and adds “صبح”, which the instruction did not say. |

Nemotron follows short Persian instructions more literally and knows the proverb. Gemma is smoother and faster, and it is the less-wrong translator of “invent dates”. Neither translation is something to ship without a glossary.

## Decision

For a Persian assistant on this hardware, run Nemotron 3.5 Bolt E4B UD-Q4_K_XL if answer correctness is the product. Budget for hybrid execution and roughly 3–12 seconds on the prompts in this sample.

Run Gemma 4 E2B if the product is interactive and the 1650 is the production GPU. It is about seven times faster end-to-end on these 40 calls (28.6 s vs 205.7 s) and it wins paraphrase detection. It is not the better reader or QA model here.

Do not use either raw checkpoint as an aspect-sentiment classifier. That task needs a constrained label head or a verifier. The 0.50 tie is the evidence.

## Reproduce

Python 3.10+ standard library only. Point the client at an OpenAI-compatible llama.cpp server that has already loaded the checkpoint you want to score.

```bash
python eval_fa.py --base-url http://127.0.0.1:8020 --model-name nemotron-ud-q4-k-xl
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
- `results/gemma-4-e2b.json` — per-item Gemma predictions
- `results/qual-nemotron.json`, `results/qual-gemma.json` — the five writing probes
