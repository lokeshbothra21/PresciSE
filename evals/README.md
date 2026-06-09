# RAGAS evaluation for PresciSE

Measures answer + retrieval quality with [RAGAS](https://docs.ragas.io). Because
RAGAS's stable line needs the old langchain 0.2.x ecosystem (incompatible with
the app's langchain 1.x), it runs in its **own venv**, decoupled from the app via
a predictions file:

```
STEP 1 (main venv)            STEP 2 (.venv-ragas)
generate_predictions.py  ──►  predictions.jsonl  ──►  score_ragas.py  ──►  scores
   (runs the agent)                                   (RAGAS + Gemini judge)
```

## One-time setup of the eval venv
```bash
python3 -m venv .venv-ragas
.venv-ragas/bin/pip install -r evals/requirements-ragas.txt
```

## Run it (functional check against the bundled gromacs PDF)
```bash
# STEP 1 — main venv. Seed a PDF + run the example questions through the agent.
python evals/generate_predictions.py \
    --testset evals/testset.example.json \
    --seed-pdf data/pdfs/<the-gromacs-pdf>.pdf \
    --user evaluser

# STEP 2 — eval venv. Score the predictions (needs GEMINI_API_KEY).
.venv-ragas/bin/python evals/score_ragas.py --predictions evals/predictions.jsonl
```
Output: averaged scores in the console + per-question scores in `evals/ragas_scores.csv`.

> Notes: `--seed-pdf` writes chunks into the DB (re-running re-seeds under a new
> doc_id → duplicates; delete the eval user's docs or use a separate
> `PRESCISE_DATABASE_URL` for eval). The judge adds Gemini calls on top of the
> agent's ~11–14/query — lower `--workers` if you hit rate limits.

## The test-set format (how to author your 20 questions)

A test set is a JSON list. Each item:

```json
{
  "question": "A question answerable from the document(s).",
  "ground_truth": "A concise correct reference answer (optional but recommended).",
  "user_id": "evaluser"   // optional; defaults to --user
}
```

- **`question`** (required) — what you'd actually ask.
- **`ground_truth`** (optional) — a short, factual reference answer. **With it** you
  get the strongest metrics (`context_recall`, `answer_correctness`); **without it**
  you still get `faithfulness`, `answer_relevancy`, `context_precision`.
- **`user_id`** (optional) — evaluate against a specific user's documents.

### How to create a good 20-question set
1. **Pick the document(s)** you'll evaluate and upload/seed them for one `user_id`.
2. **Write questions straight from the source** so a correct answer provably
   exists in the text. Aim for a mix:
   - factual lookups ("What is X?"), 
   - explanatory ("Why/how does X work?"),
   - comparative ("difference between X and Y"),
   - a few **formula** questions (this corpus is formula-heavy).
3. **Write each `ground_truth` by reading the source**, not from memory — 1–3
   sentences, only what the document supports. (Optionally note the page/section
   in your own records so you can re-check.)
4. Include **2–3 out-of-scope questions** (answerable by no uploaded doc) to check
   the system correctly declines instead of hallucinating — low faithfulness on a
   confident wrong answer is exactly what you want to catch.
5. Keep it balanced (~15 in-scope, a few hard, a few out-of-scope). 20 is plenty
   to start; grow it as you find failure cases.

### What the metrics mean
| Metric | 0→1, higher is better | Needs `ground_truth` |
|---|---|---|
| faithfulness | answer is grounded in retrieved context (no hallucination) | no |
| answer_relevancy | answer addresses the question | no |
| context_precision | retrieved chunks are relevant / well-ranked | no |
| context_recall | retrieval fetched all the info needed | yes |
| answer_correctness | answer matches the reference | yes |

Treat scores as **directional** (the judge is itself an LLM) — inspect the
low-scoring rows in `ragas_scores.csv` to see *why*.
