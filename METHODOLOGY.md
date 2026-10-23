# Fine-Tuning a Compact Multilingual Embedding Model for Portuguese (pt-BR) Product Retrieval

**Goal.** Quantify how much task- and language-specific fine-tuning changes the retrieval
quality of a small, open, self-hostable sentence-embedding model
(`intfloat/multilingual-e5-small`, 118M parameters, 384 dimensions) on
Brazilian-Portuguese product search. Amazon Titan Text Embeddings V2 (1,024 dimensions)
is measured alongside it as a general-purpose reference point, to put the size of the
fine-tuning gain in context.

Embedding quality is task- and language-specific, so none of what follows is a general
statement about one model being better than another. It is a measurement of one task
(product retrieval), in one language (pt-BR), on one dataset.

All experiments were run in AWS region `us-east-1`. Embeddings, similarity search,
and metric computation are exact (brute-force cosine over L2-normalized vectors,
NumPy) — no approximate-nearest-neighbour index — so model comparisons are not
confounded by ANN approximation error.

---

## 1. Original dataset

We use the **Amazon ESCI "Shopping Queries Dataset"** (Reddy et al., 2022), a
large-scale benchmark for semantic product search. Each row is a `<query, product>`
judgement annotated with an ESCI relevance label:

- **E** — Exact (fully relevant)
- **S** — Substitute (acceptable alternative)
- **C** — Complement (complementary item)
- **I** — Irrelevant

The dataset is multilingual (English/US, Spanish/ES, Japanese/JP) and ships as three
files:

| File | Rows | Key columns |
| --- | --- | --- |
| `shopping_queries_dataset_examples.parquet` | 2,621,288 | example_id, query, query_id, product_id, product_locale, esci_label, small_version, large_version, split |
| `shopping_queries_dataset_products.parquet` | 1,814,924 | product_id, product_title, product_description, product_bullet_point, product_brand, product_color, product_locale |
| `shopping_queries_dataset_sources.csv` | 130,652 | query_id, source |

---

## 2. Sampling from the original data

We restricted the study to the **US (English) locale** and to the **E / S / I**
labels (Complement was dropped). After this filter:

- Judgement rows: **1,778,776**
- Unique queries: **97,345**
- Label distribution: E 1,247,558 · S 369,313 · I 161,905

### 2.1 Per-query composition target

To obtain a controlled benchmark we required each selected query to contain a
specific mix of labels. The initial strict rule (E = 1–2, S = 0–2, I = 4–7) yielded
only **103** eligible queries; a relaxed rule (E = 1–4, S = 0–2, I = 2–7) yielded
only **286** — both far below the target of 10,000. The dataset is dominated by
Exact matches (avg. ~18–20 judgements/query), so any hard *upper* cap rejects almost
every query.

### 2.2 Trim-based sampling (final method)

Instead of rejecting queries with too many judgements, we **trim** them. A query is
*eligible* if it can reach the per-label **minimums** (E ≥ 1, I ≥ 2; S has no
minimum); for each selected query we then randomly keep a target count per label
drawn within its range (E 1–4, S 0–2, I 2–7), capped by availability.

- Eligible queries (could reach minimums): **23,238**
- Queries sampled (seed = 42): **10,000**
- Resulting judgement rows after trimming: **62,689**
- Distinct products referenced: **59,270**
- Label distribution in sample: **E 22,464 · I 33,947 · S 6,278**

All randomness uses a fixed seed (42) for both the query draw and the per-label
trimming, so the sample is fully reproducible.

### 2.3 Exported sample (pre-translation)

| File | Records | Fields |
| --- | --- | --- |
| `sample_queries.jsonl` | 62,689 | example_id, query |
| `sample_products.jsonl` | 59,270 | product_id, product_title, product_description, product_bullet_point, product_color |

Character counts of the exported files: queries 3,508,416 chars; products
69,839,463 chars.

### 2.4 Filtered-catalog descriptive statistics

For the 59,270 distinct products in the sample: 26,519 unique brands (~2.2 products
per brand) and 11,769 unique colours (11,462 after lowercase normalization).
Attribute coverage: title 100%, bullet_point 80.5%, brand 89.2%, colour 61.9%,
description 47.8%.

---

## 3. Translation to Portuguese (pt-BR) with Amazon Translate

We translated both products and queries from English to Portuguese (Brazil) using
**Amazon Translate** (neural MT), validated against AWS documentation:

- **Mode:** asynchronous **batch** translation (`StartTextTranslationJob`), chosen
  over real-time for the full corpus. Pricing is per source character ($15 / million
  for standard and batch), so mode affects throughput, not unit cost.
- **Format:** Amazon Translate batch does **not** accept JSONL. We therefore wrote
  the **field values** into `.xlsx` documents (translated cell-by-cell with sheet
  structure preserved), then reassembled JSONL from the output. Each batch document
  is capped at **1 MB of translatable text**, so the corpus was chunked.
- **Documents submitted:** **81** total — 80 product chunks + 1 query file.
- **Records translated:** **69,270** — 59,270 products + **10,000 unique queries**
  (queries were **deduplicated** before translation; the 62,689 judgement rows
  repeat the same query strings, so translating uniques avoided ~52k redundant
  calls).

### 3.1 Custom terminology (brand preservation)

Out-of-the-box translation mistranslated brand-like tokens (e.g. a query *"revent"*
became *"evitar"*). We built an Amazon Translate **custom terminology** file from the
catalog brands — **26,503** brand terms (after dropping 168 junk/placeholder values),
each mapped to itself (en→pt, UNI directionality) so brands are left untranslated.
Imported as terminology `product-brands` and applied to all jobs.

This reliably preserved in-catalog brands (Panasonic, Aero Pure, Delta Electronics,
OtterBox, …). It does **not** fix query-side tokens that are not actual catalog
brands or that differ in case (custom terminology is case-sensitive); these were
treated as best-effort.

### 3.2 Cost and integrity

- **Billable characters:** 63,155,306 (product fields 62,926,077; unique queries
  223,224; xlsx headers 6,005) → **≈ US$947** at $15/million.
- **Job result:** 81/81 documents translated, **0 errors**.
- **Reassembly integrity:** product IDs matched exactly (59,270 in / 59,270 out);
  **0 fields lost** across title/description/bullet_point/colour; all 62,689 query
  rows received a translation (the 10k unique translations were expanded back over
  every judgement row by matching the English query string).

Output files: `sample_products_ptbr.jsonl` (59,270), `sample_queries_ptbr.jsonl`
(62,689).

### 3.3 Joined labelled dataset

We joined translated queries (via `example_id`) and products (via `product_id`) with
the ESCI labels to produce one record per judgement:

- **`sample_dataset_ptbr.jsonl` — 62,689 records**
- Fields: `example_id, query_id, query_en, query_pt, product_id, esci_label, split,
  product_title_pt, product_description_pt, product_bullet_point_pt,
  product_color_pt`

---

## 4. Train / test / validation split

We replaced the original ESCI split with a fresh, reproducible **query-level**
70 / 20 / 10 split (seed = 42). Splitting at the query level keeps every judgement of
a query inside a single partition, preventing query leakage between train and test.

| Split | Unique queries | Judgement rows |
| --- | --- | --- |
| train | 7,000 (70%) | 43,989 |
| test | 2,000 (20%) | 12,487 |
| validation | 1,000 (10%) | 6,213 |
| **Total** | **10,000** | **62,689** |

Verified query overlaps: train∩test = 0, train∩val = 0, test∩val = 0. (Product
overlap across splits is expected and harmless for retrieval, since the model never
sees test *queries* during training.)

> Note: this replaces the official ESCI split, so absolute numbers are not directly
> comparable to published ESCI leaderboard results.

---

## 5. Models compared

| Role | Model | Params | Dims | Notes |
| --- | --- | --- | --- | --- |
| Model we fine-tune | `intfloat/multilingual-e5-small` | 118M | 384 | retrieval-pretrained (contrastive), `query:`/`passage:` prefixes, multilingual incl. pt, MIT license, self-hostable |
| General-purpose reference point | Amazon Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) | undisclosed | 1,024 | 8,192-token context, multilingual incl. Portuguese, managed on Amazon Bedrock, no training required |

The two models play different roles here. One is specialized for this task and language
using labeled domain data; the other is a managed, general-purpose model that produces
multilingual representations with no training effort, and it is included to show what the
fine-tuning gain is worth relative to a competent off-the-shelf starting point. This is
not a parameter-count comparison — Titan's size is not public.

## 6. Evaluation methodology

For each query the model embeds the query and the product passages, ranks products
by cosine similarity (L2-normalized → dot product), and we score the ranking against
the ESCI labels.

- **Product passage text:** `product_title_pt` + `product_bullet_point_pt` +
  `product_description_pt` (identical input for every model).
- **Graded relevance gains (ESCI / KDD-Cup convention):** E = 1.0, S = 0.1,
  I = 0.0. Unjudged products retrieved from the full catalog count as gain 0.
- **Metrics:** nDCG@k, Recall@k (k ∈ {5, 10}), and MRR (rerank mode). Definitions:
  - *Recall@k* = (relevant E products retrieved in top-k) / (total E for the query).
  - *nDCG@k* = DCG@k / IDCG@k, with DCG@k = Σ gainᵢ / log₂(i+1); IDCG is the ideal
    ranking of all the query's judged gains. Order- and grade-aware; the headline
    metric.
  - *MRR* = mean of 1/rank of the first relevant (E) item.
- **Two retrieval regimes:**
  - **rerank** — rank only a query's already-judged candidates (ESCI Task-1 style).
    Easy, high ceiling (~0.83 nDCG), poor at separating models.
  - **corpus (full catalog)** — retrieve from **all 59,270 products**. Realistic
    product search; the regime used for all headline results below.

For e5, `query:` / `passage:` prefixes are applied (as the model expects). The base
and fine-tuned e5 models were evaluated **identically** (local inference, max
sequence length = 256 tokens), so the only difference between them is the learned
weights. Titan used its own tokenizer and 8,192-token context (i.e. it saw *more*
product text, if anything an advantage to the baseline).

> **Metric caveat.** Only the products judged for a query are labelled (~6 per
> query); the other ~59k are unlabelled and scored as irrelevant. A genuinely
> relevant but unjudged product therefore counts as a miss, which deflates the
> absolute Recall/nDCG values. This penalty applies identically to every model, so
> model-vs-model comparison remains fair; the absolute numbers are not a true recall.

---

## 7. Reference-point metrics (full corpus, test split)

Test split: **2,000 queries**, retrieving from **all 59,270 products**. Neither model has
seen this task or this language-specific data.

| Model | dims | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| e5-small (base, no fine-tuning) | 384 | 0.2494 | 0.2764 | 0.2795 | 0.3452 |
| Titan Text Embeddings V2 | 1024 | 0.3404 | 0.3730 | 0.3800 | 0.4580 |

Without any training, Titan Text Embeddings V2 scores ~0.10 nDCG@10 above the untuned
e5-small on this task, which is the practical value of a managed general-purpose model:
usable retrieval quality with no data work. It also sets the bar the fine-tune is measured
against. (In the *rerank* regime the two were nearly tied, ~0.83 nDCG@10, illustrating why
that regime is uninformative for model selection.)

---

## 8. Fine-tuning the base model

The base e5-small was fine-tuned on **Amazon SageMaker** (training jobs
auto-terminate, so no idle cost).

### 8.1 Training data construction (train split only)

Contrastive triplets `(anchor, positive, hard-negative)`:

- **Positive:** a query's Exact (E) product.
- **Hard negative:** one of the same query's Irrelevant (I) products (randomly
  chosen, seed = 42).
- **Substitutes (S) were excluded** from training to keep the E-vs-I signal crisp
  (S is still scored, with gain 0.1, at evaluation time).
- e5 `query:` / `passage:` prefixes were baked into the text.

Result: **15,761 training triplets**. Validation set for in-training checkpoint
selection: **1,000 queries / 6,161 corpus documents** (InformationRetrievalEvaluator).

### 8.2 Training configuration

| Item | Value |
| --- | --- |
| Framework | sentence-transformers 2.7.0 (HF DLC: transformers 4.36, PyTorch 2.1, py310) |
| Loss | `MultipleNegativesRankingLoss` (in-batch negatives + explicit hard negative) |
| Epochs | 2 |
| Batch size | 32 |
| Max sequence length | 256 tokens |
| Learning rate | 2e-5, warmup ratio 0.1 |
| Precision | AMP (fp16) |
| Instance | `ml.g5.xlarge` (1× NVIDIA A10G, 24 GB) |
| Checkpoint selection | best validation nDCG@10 |

An initial attempt OOM'd at batch 64 / 512 tokens (192 sequences × 512 tokens of
activations on a 24 GB GPU); reducing to batch 32 / 256 tokens plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` resolved it.

- **Billable training time:** 678 s (~11 min) ≈ **US$0.27**.
- **Validation nDCG@10 during training** (split-corpus): 0.6005 → 0.6023 (the
  split-corpus validation task is easy and near-saturated; the full-corpus test
  reveals the true gain).

---

## 9. Fine-tuned metrics and results

Identical evaluation (full corpus, test split, local inference, 256 tokens).

| Model | dims | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| e5-small (base) | 384 | 0.2494 | 0.2764 | 0.2795 | 0.3452 |
| **e5-small (fine-tuned)** | **384** | **0.3970** | **0.4320** | **0.4437** | **0.5276** |
| Titan Text Embeddings V2 | 1024 | 0.3404 | 0.3730 | 0.3800 | 0.4580 |

### 9.1 What we achieved

**(a) Large fine-tuning lift** (same architecture, dims, and input — only the weights
changed):

| Metric | Base | Fine-tuned | Absolute Δ | Relative Δ |
| --- | --- | --- | --- | --- |
| nDCG@10 | 0.2764 | 0.4320 | +0.1556 | +56.3% |
| nDCG@5 | 0.2494 | 0.3970 | +0.1476 | +59.2% |
| Recall@10 | 0.3452 | 0.5276 | +0.1824 | +52.8% |
| Recall@5 | 0.2795 | 0.4437 | +0.1642 | +58.7% |

**(b) Where the fine-tuned model lands relative to the general-purpose reference point,
on this task:**

| Metric | Fine-tuned e5-small (384d) | Titan V2 (1024d) | Δ | Relative |
| --- | --- | --- | --- | --- |
| nDCG@10 | 0.4320 | 0.3730 | +0.0590 | +15.8% |
| nDCG@5 | 0.3970 | 0.3404 | +0.0566 | +16.6% |
| Recall@10 | 0.5276 | 0.4580 | +0.0696 | +15.2% |
| Recall@5 | 0.4437 | 0.3800 | +0.0637 | +16.8% |

On this pt-BR product-retrieval task, ~US$0.27 of GPU time on labeled domain data moved a
384-dimension open model above the general-purpose reference point on the task metrics,
even though the reference point has both higher dimensionality and a longer context
window. Read this as a measure of what task- and language-specific specialization buys on
a specific task, not as a general ranking of the two models: a general-purpose model is
trained for broad coverage across many tasks and languages, which is exactly what this
narrow evaluation does not reward. The smaller embedding also implies a ~2.7× smaller
vector index (384 vs 1,024 dims), reducing storage and serving cost — though a self-hosted
model shifts inference cost and operational work onto you, which the managed option
absorbs.

---

## 10. Reproducibility, artifacts, and limitations

### Pipeline / artifacts

Every stage below lives in `reproduce_experiment.ipynb`; the "Step" column is the
notebook section that implements it. Steps 1-4 only execute when `DATA_MODE = "full"`.

| Stage | Step in the notebook | Output |
| --- | --- | --- |
| Download the original ESCI data | 1 | `esci-code/` (parquet) |
| Sampling (controlled label mix) | 2 | in-memory `df_sample`, `cat` |
| Brand custom terminology | 3.1 | `out/brand_terminology.csv`, imported as `product-brands` |
| Translation (Amazon Translate batch) | 3.2 | `s3://.../translate-out/...` |
| Reassembly | 3.3 | `prod_pt`, `q_pt` |
| Labelled dataset + query-level split | 4 | `out/dataset_ptbr.json` |
| Load the pt-BR dataset (both tracks) | 4b | `records` |
| Evaluation harness | 5 | `evaluate_corpus()` |
| Reference points (base e5, Titan) | 6 | `base_res`, `titan_res` |
| Fine-tune data prep | 7.1 | `finetune_data/` (triplets + validation IR set) |
| Training (SageMaker job) | 7.2 | `out/e5-small-ptbr-ft/` |
| Final comparison | 8 | the results table |

In `DATA_MODE = "bundled"` the notebook skips steps 1-4 and loads
`data/sample_dataset_ptbr.sample.jsonl` at step 4b instead. That sample retrieves from
3,072 products rather than 59,270, so its metrics run well above the values reported in
sections 7 and 9 — it validates the pipeline, it does not reproduce the benchmark.

The measured metric files behind sections 7 and 9 are in `artifacts/eval_results/`: three
`*_corpus_fullcorpus_*.json` files for the full-catalog results reported here, plus two
`*_rerank_*.json` files backing the ~0.83 nDCG@10 rerank figures cited above. The
notebook implements the corpus regime only; the rerank numbers are included as recorded
evidence for the claim that the rerank regime fails to separate models.

Fixed seed = 42 throughout (sampling, split, hard-negative selection).

### Approximate cost
Amazon Translate ≈ US$947; SageMaker training ≈ US$0.27; Titan/endpoint embedding for
evaluation ≈ cents. (A GPU inference endpoint, `ml.g6.2xlarge`, was used briefly to
serve the base model and was deleted afterwards.) These are the charges observed in
`us-east-1` at the time of the run; prices vary by AWS Region and change over time, so
check the current AWS pricing pages before reproducing the paid track.

### Limitations
1. **Sparse labels** deflate absolute Recall/nDCG (unjudged products scored as
   irrelevant); comparisons are relative.
2. **Custom split** replaces the official ESCI split, so results are not directly
   comparable to published ESCI numbers.
3. **Single run, single seed** — no variance/confidence intervals reported.
4. **Translation artifacts** — MT errors and query-side brand tokens not in the
   catalog remain (e.g. "revent").
5. **S excluded from training** — a substitute-as-positive variant is left as an
   ablation.
6. **Titan size undisclosed** — comparisons refer to embedding dimensionality and to its
   general-purpose, untuned role here, not to a measured parameter count.
7. **Sequence length 256** for e5; very long descriptions are truncated (most
   product text fits).
8. **Task- and language-scoped result** — the reference point is measured off the shelf on
   one narrow task in one language. Nothing here characterizes its quality on the broad
   range of tasks and languages a general-purpose model is built for.

### Suggested future work
Dimensionality sweep on Titan (256/512/1024); S-as-positive ablation; multiple
seeds with confidence intervals; comparison against `multilingual-e5-large` and
`bge-m3`; evaluation on the full 1.8M-product catalog.
