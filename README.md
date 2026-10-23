# Fine-tune a compact embedding model for language- and domain-specific product retrieval on Amazon SageMaker

Semantic product search relies on text embeddings to match what a shopper types to the
right items in a catalog. General-purpose embedding models, such as Amazon Titan Text
Embeddings on Amazon Bedrock, provide multilingual representations and are a managed
starting point that requires no training. At the same time, embedding quality is
task- and language-specific: the representation suited to ranking products against
short, abbreviated shopper queries in a given language can differ from the one suited to
general semantic similarity.

This sample shows how to specialize a compact, open embedding model for one task
(product retrieval) in one language (Brazilian-Portuguese, pt-BR) by fine-tuning it on
your own domain data with Amazon Translate, Amazon Bedrock, and Amazon SageMaker. The
approach is language-agnostic and can be reproduced for any language your catalog needs.
We use Amazon Titan Text Embeddings V2 as a general-purpose reference point to quantify
how much task- and language-specific fine-tuning adds for this use case. The complete
workflow is in a single reproducible notebook,
[`reproduce_experiment.ipynb`](reproduce_experiment.ipynb).

## Repository layout

```
README.md                    this overview: situation, task, action, results
reproduce_experiment.ipynb   the whole workflow, runnable top to bottom
requirements.txt             local Python dependencies
METHODOLOGY.md               detailed write-up: sampling, translation, metrics, results
DATASET_PROVENANCE.md        data source, license, attribution
NOTICE                       attributions + the changes made to the source dataset
LICENSE                      MIT-0, covering the sample code and documentation
data/
  sample_dataset_ptbr.sample.jsonl   500-query pre-translated pt-BR sample (4.4 MB)
  LICENSE                    Apache-2.0, covering the derived pt-BR data
artifacts/
  eval_results/*.json        raw metric files behind the results table
```

Run the notebook from the repository root: the bundled-sample path and the artifact
paths are relative to it.

The full derived pt-BR dataset (62,689 judgements / 59,270 products, ~97 MB) is not
included in this repository because of its size. Regenerate it with
`DATA_MODE = "full"`, which reproduces it deterministically from the original English
ESCI data with a fixed seed.

**How `data/sample_dataset_ptbr.sample.jsonl` was produced.** It is a slice of that full
dataset, cut by `query_id` with a fixed seed (42) rather than by row, so every selected
query keeps its complete set of E/S/I judgements and the ranking metrics stay
well-defined. 500 queries were drawn, split-proportionally (350 train / 100 test / 50
validation), preferring queries that carry both an Exact and an Irrelevant judgement so
they can form a training triplet. The product passage is the pt-BR title, bullet points,
and description concatenated — the same composition the notebook uses everywhere.

## Situation

Embedding models power semantic retrieval. General-purpose models are trained for broad
coverage across many tasks and languages, which is useful when you need a model with no
training effort. For a narrow task such as product retrieval in a specific language,
however, the domain vocabulary, brand names, and the short or abbreviated queries typical
of e-commerce may benefit from representations tuned to that task and language. Vector
dimensionality also matters operationally: larger vectors increase index size, memory,
and serving cost.

A large, human-annotated product-search benchmark already exists: the **Shopping Queries
Dataset (ESCI)**, released publicly by Amazon Science
([`amazon-research/esci-code`](https://github.com/amazon-research/esci-code),
Apache-2.0, arXiv:2206.06588). It pairs customer queries with candidate products graded
as **E**xact, **S**ubstitute, **C**omplement, or **I**rrelevant. Its text is in English,
Spanish, and Japanese, so we translate a subset to pt-BR to demonstrate the workflow.

## Task

We answer a practical question for a pt-BR retail use case: **how much can task- and
language-specific fine-tuning improve a compact, open embedding model on product
retrieval?**

- **Model we fine-tune:** `intfloat/multilingual-e5-small` (118M parameters, 384-dimension
  embeddings, MIT license, retrieval-pretrained, multilingual including Portuguese).
- **General-purpose reference point:** Amazon Titan Text Embeddings V2 on Amazon Bedrock
  (1,024 dimensions, 8,192-token context, multilingual) — a managed model used to
  contextualize the fine-tuning gains.
- **Evaluation:** full-catalog retrieval on a held-out test set, measured with nDCG@k and
  Recall@k using the ESCI relevance grades.

Because embedding suitability depends on the task and language, this comparison is not
about one model being better than another in general. It illustrates how much
specializing a model for a specific task and language changes results on that task.

## Action

The end-to-end workflow, implemented in `reproduce_experiment.ipynb`:

1. **Get the data.** Download the ESCI Shopping Queries Dataset from Amazon Science.
2. **Sample.** Select a reproducible, controlled subset of **10,000 queries** (US
   locale, E/S/I labels) with a balanced per-query label mix.
3. **Translate to pt-BR with Amazon Translate.** Run a batch translation job on the
   product and query text, using an Amazon Translate **custom terminology** built from
   the catalog brands so brand names are preserved rather than translated. (Swap the
   target language here to reproduce the workflow for any other language.)
4. **Build the dataset and split.** Join the translations with the ESCI labels and
   create a query-level 70/20/10 train/test/validation split (no query leakage).
5. **Build the evaluation harness.** Define the graded-relevance metrics and the
   full-catalog retrieval loop used identically for every model.
6. **Establish reference points.** Embed queries and the full product catalog with the
   base e5-small model (locally) and with Amazon Titan Text Embeddings V2 (on Amazon
   Bedrock), and evaluate both on full-corpus retrieval.
7. **Fine-tune on Amazon SageMaker.** Fine-tune e5-small with
   `MultipleNegativesRankingLoss` (Exact products as positives, Irrelevant products as
   hard negatives) as a SageMaker training job that terminates automatically.
8. **Compare.** Re-evaluate the fine-tuned model with the identical harness.

The numbering matches the section headings in the notebook, so step *n* here is section
*n* there.

### Two ways to run the notebook

The first cell of `reproduce_experiment.ipynb` has a `DATA_MODE` switch that selects
which of two tracks you take. Both converge on the same variable (`records`) and share
steps 5 onward, so you only choose how the pt-BR data comes into existence.

| | `DATA_MODE = "bundled"` (default) | `DATA_MODE = "full"` |
| --- | --- | --- |
| pt-BR data | pre-translated sample included here | you generate it |
| Steps 1-4 | skipped | executed |
| Amazon Translate | not called | **~US$947** for 10,000 queries |
| Data prep cost | free | paid |
| Retrieval corpus | 3,072 products | 59,270 products |
| Reproduces the results table | no (see note) | yes |

**Bundled (start here).** `data/sample_dataset_ptbr.sample.jsonl` holds a 500-query
slice of the derived pt-BR dataset (3,088 judgements, 3,072 products, 4.4 MB), cut by
query so each query keeps its full E/S/I candidate set. It lets you run the whole
workflow and see it work without paying for translation. Every cell in steps 1-4
carries a banner marking it as `"full"`-only and prints a skip message otherwise, so
"Run All" never starts a translation job by accident. The only paid step left on this
track is the SageMaker fine-tuning job in step 7.2 (~US$0.27); it stops with an
actionable message if `S3_BUCKET` is still unset, and `RUN_TRAINING = False` skips it
outright.

> **The bundled sample will not reproduce the numbers in the results table.** It
> retrieves from 3,072 products instead of 59,270. A smaller corpus is a much easier
> task, so nDCG and Recall come out considerably higher (base e5-small scores ~0.47
> nDCG@10 on the sample versus 0.2764 on the full catalog). The sample validates the
> pipeline; it is not a benchmark.

**Full.** Set `DATA_MODE = "full"` to regenerate the pt-BR data from the original
English ESCI: clone the dataset, sample `N_QUERIES`, translate with Amazon Translate,
and reassemble. The one cell that spends money (step 3, starting the batch job) has a
second, independent guard — `PROCEED_WITH_PAID_TRANSLATION = False` — that you must
flip by hand, and it prints a character count and cost estimate for your specific
`N_QUERIES` before doing anything. Lower `N_QUERIES` for a cheap trial.

### Use your own data instead of the translated sample

The translated ESCI sample is only a stand-in for real catalog data. If you already have
queries and products in your target language with relevance judgements (for example,
from search logs or human annotation), you can skip translation entirely and use them
instead. Keep `DATA_MODE = "bundled"` and point `BUNDLED_SAMPLE` in the first cell to
your own JSONL file, with one judgement per line in the same schema as
`data/sample_dataset_ptbr.sample.jsonl`:

```json
{"query_id": 9, "query_pt": "...", "product_id": "B07L4383XR", "product_pt": "...", "esci_label": "E", "split": "train"}
```

- `query_pt` / `product_pt` hold the query and product text in **your** language; the
  `_pt` suffix is just the field name the notebook reads.
- `esci_label` is `E` (exact), `S` (substitute), or `I` (irrelevant).
- `split` is `train`, `test`, or `validation`, assigned per query so that every
  judgement of a query lands in the same split.
- Use one consistent type for `query_id` (all integers or all strings).
- Only queries that have both an `E` and an `I` judgement produce training triplets,
  and test queries without an `E` are left out of Recall@k, so make sure most of your
  queries have both.

Steps 5 onward then run unchanged on your data. Your scores will not be comparable to
the results table below, because the catalog, queries, and labels are different.

### Prerequisites

- Python 3.9+ with the packages in `requirements.txt`.
- For fine-tuning (step 7): an AWS account with Amazon SageMaker access, an S3 bucket,
  and a SageMaker execution role (auto-detected inside SageMaker Studio). Set
  `RUN_TRAINING = False` in the first cell to skip it.
- For the Titan reference point (step 6): Amazon Bedrock with Titan Text Embeddings V2
  enabled. Optional — set `RUN_TITAN = False`, or let the cell report the access error
  and continue; either way the comparison table omits that row.
- Only for `DATA_MODE = "full"`: Amazon Translate plus an IAM role it can assume to
  read and write your S3 bucket, and a working **`git-lfs`** client. The ESCI `.parquet`
  files are Git LFS objects: without git-lfs the clone still succeeds but writes ~130-byte
  pointer stubs instead of the data. Step 1 checks for this and stops with instructions
  rather than failing later on an unreadable parquet.

Scope the IAM roles you create to the S3 prefixes and the specific Amazon Translate,
Amazon Bedrock, and Amazon SageMaker actions this workflow calls, rather than reusing a
broad administrative role.

### Walk through the solution

```bash
pip install -r requirements.txt
jupyter lab reproduce_experiment.ipynb
```

> **On `sentence-transformers` versions.** `requirements.txt` covers your local
> environment, where the notebook only calls `model.encode()`. The SageMaker training job
> installs its own pin (`sentence-transformers==2.7.0`), which the notebook writes into
> `finetune_src/requirements.txt`, because the training script uses the `model.fit()` API
> from the 2.x line. The two are intentionally independent; leave them that way.

Run the cells top to bottom. In the default bundled mode, steps 1-4 skip themselves and
nothing before step 6 touches AWS. Set `REGION`, `S3_BUCKET` and `SAGEMAKER_ROLE_ARN`
in the first cell before the fine-tuning step; `TRANSLATE_ROLE_ARN` is only required in
`"full"` mode. All logic — sampling, the Amazon Translate batch job, reassembly, the
split, evaluation, and the SageMaker `train.py` — is inline in the notebook.

### Costs

You are responsible for the AWS charges incurred by running this sample.

| Step | Approximate cost |
| --- | --- |
| Amazon Translate, `DATA_MODE = "full"`, 10,000 queries | ~US$947 (63.2M characters at US$15/million) |
| Amazon SageMaker fine-tuning job (`ml.g5.xlarge`, ~11 min) | ~US$0.27 |
| Amazon Bedrock embeddings for evaluation | cents |
| `DATA_MODE = "bundled"` data preparation | none |

Translation is the only expensive step, and it only runs in `"full"` mode. These are the
charges observed in `us-east-1` at the time of the run; prices vary by AWS Region and
change over time, so check the current [AWS pricing pages](https://aws.amazon.com/pricing/)
before you run the paid track.

### How we measure retrieval quality

For each query, we embed the query and the product passages, rank products by cosine
similarity (L2-normalized vectors, so cosine equals a dot product), and score the
ranking against the ESCI labels with graded gains (E=1.0, S=0.1, I=0.0). We report
**nDCG@k** (order- and grade-aware ranking quality) and **Recall@k** (coverage of
relevant products), for k = 5 and 10, retrieving from the full catalog of 59,270
products.

> Note: only the products judged for each query are labeled, so unjudged products count
> as irrelevant. This deflates the absolute scores equally for every model, so the
> comparison remains consistent.

## Results

Full-corpus retrieval on the held-out test set (2,000 queries, 59,270 products):

| Model | Dimensions | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| e5-small (base) | 384 | 0.2494 | 0.2764 | 0.2795 | 0.3452 |
| **e5-small (fine-tuned)** | **384** | **0.3970** | **0.4320** | **0.4437** | **0.5276** |
| Titan Text Embeddings V2 | 1024 | 0.3404 | 0.3730 | 0.3800 | 0.4580 |

These are measured values from a single run on the full 10,000-query dataset; the raw
metric files are in [`artifacts/eval_results/`](artifacts/eval_results). Re-running the
`"full"` track will land close to but not exactly on these figures: the sampling and the
translation are separate executions, and it is one seed with no confidence intervals.
Treat the relative gaps as the result, not the fourth decimal. The full write-up is in
[`METHODOLOGY.md`](METHODOLOGY.md).

Key observations, specific to this task (pt-BR product retrieval) and this evaluation:

- **Task- and language-specific fine-tuning increased the task metrics.** With the same
  architecture, dimensionality, and input, nDCG@10 increased from 0.2764 to 0.4320 (a 56%
  relative increase) and Recall@10 from 0.3452 to 0.5276 (a 53% relative increase),
  reflecting adaptation to the target task and language using labeled domain data.
- **The general-purpose reference point produced the results shown without training.**
  Amazon Titan Text Embeddings V2 provides multilingual representations out of the box and
  is a fit when you need coverage without training. On this product-retrieval task in
  pt-BR, the fine-tuned compact model scored higher on the task metrics and produced
  vectors roughly 2.7 times smaller (384 vs 1,024 dimensions), which reduces index storage
  and serving cost.

The takeaway is not that one model is better in general; specializing a compact model for
a specific task and language, using labeled data you likely already have, changed results
on that task by the margins above. The fine-tuning job cost about US$0.27, so periodic
re-tuning on fresh catalog data is low-cost.

## Clean up

To avoid ongoing charges, delete any resources you created:

- Delete any SageMaker inference endpoint used to serve the model (SageMaker training
  jobs stop billing automatically when they complete, but endpoints bill until deleted).
- Remove the S3 objects created for the translation job and the training data if you no
  longer need them.
- Delete the Amazon Translate custom terminology (`product-brands`) if you no longer need
  it.

## Conclusion

Amazon Titan Text Embeddings is a general-purpose, multilingual option that runs without
training. When you have a focused task, such as product retrieval in a specific language,
and labeled domain data, fine-tuning a compact open model on Amazon SageMaker can
specialize it for that task and language, improving task-specific relevance while keeping
the serving footprint small. We demonstrated this for pt-BR, and the same workflow
reproduces for any language by changing the Amazon Translate target. Adapt the notebook
to your own catalog, and consider a dimensionality sweep, multiple seeds for confidence
intervals, or evaluating on your full production catalog.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License and attribution

Two licenses apply in this repository:

- The sample code and documentation are licensed under the **MIT-0** License; see
  [`LICENSE`](LICENSE).
- The derived pt-BR dataset in `data/` is licensed under the **Apache License 2.0**,
  inherited from the source dataset; see [`data/LICENSE`](data/LICENSE) and
  [`NOTICE`](NOTICE). `NOTICE` carries the required attributions and the itemized list of
  modifications made to the original data; keep both files with any copy or derivative
  of that data you distribute.

This work builds on the **Shopping Queries Dataset (ESCI)** by Amazon Science
([`amazon-research/esci-code`](https://github.com/amazon-research/esci-code),
Apache-2.0, arXiv:2206.06588). The pt-BR text is **machine-translated** with Amazon
Translate and is not human-reviewed; it may contain errors. The base model,
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
is MIT-licensed and is downloaded at run time rather than redistributed here.
[`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md) has the longer discussion of what was
subset, translated, and re-split.

```bibtex
@article{reddy2022shopping,
  title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
  author={Chandan K. Reddy and Llu\'is M\`arquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
  year={2022}, eprint={2206.06588}, archivePrefix={arXiv}
}
```
