# Dataset Provenance & Attribution — pt-BR Shopping Queries (derivative)

## Where the data comes from

This work is built on the **Shopping Queries Dataset (ESCI)**, a large-scale
benchmark for semantic product search released publicly by **Amazon Science**.

- **Source repository:** `amazon-research/esci-code`
  (https://github.com/amazon-research/esci-code)
- **Paper:** *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
  Product Search* — Reddy, Màrquez, Valero, Rao, Zaragoza, Bandyopadhyay, Biswas,
  Xing, Subbian (2022), arXiv:2206.06588.
- **License:** Apache-2.0.

The original dataset provides customer search queries paired with candidate products
and human relevance judgements using the **ESCI** scheme — **E**xact, **S**ubstitute,
**C**omplement, **I**rrelevant. It is multilingual (English, Spanish, Japanese) and
was created to foster research on ranking and semantic matching of queries to
products.

### Citation (please cite the original authors)

```bibtex
@article{reddy2022shopping,
  title={Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
  author={Chandan K. Reddy and Llu\'is M\`arquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
  year={2022},
  eprint={2206.06588},
  archivePrefix={arXiv}
}
```

## What was used (a subset)

The full dataset was **not** used. We selected a reproducible subset (fixed random
seed) restricted to the **US / English locale** and to the **E, S, I** labels
(Complement excluded), with a controlled per-query label composition:

- **10,000 queries**
- **59,270 distinct products**
- **62,689 query–product judgements**

## What was created (a Portuguese derivative)

Using **Amazon Translate** (neural machine translation), we generated a
**Brazilian-Portuguese (pt-BR)** version of this subset, translating product text
(title, description, bullet points, colour) and the queries from English to pt-BR.

To preserve brand names during translation — which generic MT tends to mistranslate
(e.g. a brand token being turned into a common Portuguese word) — we built an **Amazon
Translate custom terminology** from the catalog brands (**26,503 brand terms**, each
mapped to itself) so brand names are carried through unchanged.

The ESCI relevance labels, query IDs, and product IDs are preserved from the original
dataset; only the natural-language text is translated. This keeps the derivative
fully aligned with the original annotations.

**Important caveat:** the Portuguese text is **machine-translated** and not
human-reviewed. It may contain translation errors, and query-side tokens that are not
actual catalog brands (or that differ in case) are translated best-effort. It should
be treated as a machine-generated derivative, not a gold human translation.

## Use and redistribution

This repository publishes the workflow that produces the pt-BR derivative, together with a
500-query sample of it (`data/sample_dataset_ptbr.sample.jsonl`) so the notebook can be run
end to end without re-translating anything. The full derived dataset is not included here
because of its size; `DATA_MODE = "full"` regenerates it deterministically from the
original English ESCI data.

Because the source is Apache-2.0 licensed, redistribution and modified/derivative
versions are permitted, provided that:

- the original work is **attributed** (authors, paper, and source repository above);
- the **Apache-2.0 license and notices are retained** and included with the
  derivative;
- the changes are **clearly stated** — here: a US/English subset was selected and
  its text machine-translated to pt-BR with Amazon Translate using a brand custom
  terminology, while preserving IDs and ESCI labels.

Accordingly, this repository credits Amazon Science and the dataset authors, links to the
original repository and paper, ships the Apache-2.0 license text (`data/LICENSE`) and the
itemized list of modifications (`NOTICE`), and labels the pt-BR data as a
machine-translated derivative with the caveats above. Keep `data/LICENSE` and `NOTICE` with
any copy or derivative of this data that you redistribute. The sample code and
documentation in this repository are licensed separately under MIT-0 (root `LICENSE`).

> Note: this is a practical attribution/licensing summary, not legal advice. If the
> derivative is published widely, confirm the exact license/notice requirements and
> Amazon Translate's service terms for redistributing translated output.
