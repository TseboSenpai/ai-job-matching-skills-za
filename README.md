# AI-Based Job Matching and Skills Development in Local Economic Contexts

An AI and machine learning job-matching system that uses local knowledge to recommend fairer, more relevant jobs and training to job seekers in Limpopo, South Africa.

## Overview

Most job-matching systems are built on national or global data. They do not recognise local qualifications, ignore how far people can travel, and suggest training that is not available nearby. This project tests whether adding local labour-market knowledge improves AI job recommendations.

One matching algorithm is run in two modes:

| | Generic mode | Local-aware mode |
|---|---|---|
| Semantic similarity (sentence-transformer `all-MiniLM-L6-v2`) | Yes | Yes (identical) |
| Skill overlap (Jaccard) | ESCO skills | ESCO skills + South African vocabulary |
| Credential fit with equivalence rules (e.g. NC(V) Level 4 = Grade 12) | No | Yes |
| Location and mobility (reachable districts) | No | Yes |

Both modes are evaluated on 427 real Limpopo job adverts and 600 synthetic job seekers with known correct matches, using NDCG@k, precision@k and recall@k, paired Wilcoxon tests and bootstrap confidence intervals. Fairness is assessed across gender, age, education level and district, and missing credentials are matched to training routes available in Limpopo.

## Main results (test set, 480 candidates)

| Measure | Generic | Local-aware |
|---|---|---|
| NDCG@10 | 0.243 | 0.348 |
| Precision@10 | 0.104 | 0.193 |
| Recall@10 | 0.221 | 0.489 |

- 81% of the Limpopo adverts mention South African credentials or workplace terms, and 92% of these are missing from ESCO.
- Under family-only relevance the modes perform similarly (NDCG@10 0.293 vs 0.314), so the gain does not reduce occupational relevance.
- Job seekers below matric or with TVET or trade qualifications benefit most.
- 92% of top-three recommendations are usable directly or after affordable, reachable local training.

The candidates are synthetic, so these results describe how the system behaves under controlled conditions, not real hiring outcomes.

## Repository structure

```
notebooks/
  01_data_collection.ipynb       data loading, cleaning and district tagging
  02_local_vocabulary.ipynb      local credentials, equivalence rules, ESCO coverage
  03_synthetic_candidates.ipynb  job families, requirements, candidates, ground truth
  04_matching.ipynb              skill extraction, embeddings, generic vs local-aware scores
  05_evaluation.ipynb            metrics, significance tests, sensitivity, fairness, training pathways
src/
  limpopo.py                     shared functions (credential matching, requirement parsing,
                                 district reachability, ranking metrics)
data/
  raw/
    esco/                        ESCO v1.2.1 CSV files (not included, see below)
    local/                       local vocabulary, equivalence rules, job families, training options
    ...                          Limpopo job adverts
  processed/                     created by the notebooks
results/                         tables created by the notebooks
figures/                         charts created by the notebooks
```

## How to run

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download ESCO v1.2.1 from https://esco.ec.europa.eu/en/use-esco/download (language: English, format: CSV) and unzip the CSV files into `data/raw/esco/`.

3. Open the notebooks (for example in VS Code or with `jupyter notebook`) and run them in order, 01 to 05. Each notebook reads the outputs of the previous one. The first run downloads the sentence-transformer model (about 90 MB).

Random steps use a fixed seed (42), and the settings of each run are saved in `data/processed/candidates_config.json` and `matcher_config.json`.

## Data sources

- Job adverts: Limpopo posts from DPSA Public Service Vacancy Circulars 28 to 32, 34, 36 and 38 of 2026 (https://www.dpsa.gov.za/newsroom/psvc/), and Limpopo adverts from PNet. Descriptions are structured summaries of the adverts, not the full text.
- ESCO v1.2.1, European Commission (https://esco.ec.europa.eu). ESCO is not redistributed here.
- Local vocabulary, equivalence rules and training options: compiled by the author from job adverts and official sources (listed in the `source` column of each file). Rules and costs marked `verified = 0` are assumptions or estimates.

No personal data is used. All job seekers are synthetic.

## Licence

Copyright © 2026 Tsebo Molope. All rights reserved.
This code is shared for viewing and academic assessment only. It may not be copied, modified or reused without the author’s written permission.
