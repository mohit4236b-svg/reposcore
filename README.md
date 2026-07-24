# reposcore


# ⭐ RepoScore: GitHub Repository Quality Predictor

RepoScore predicts whether a GitHub repository is likely to be a "high-quality" project based on its metadata, README content, and topic tags. Enter any public GitHub repo and get an instant quality prediction, powered by a Random Forest classifier trained on ~2,200 real repositories.

**Live demo:** paste a repo (e.g. `scikit-learn/scikit-learn`) into the Streamlit app and get a prediction with confidence score.

---

## What "quality" means here

There's no built-in "quality" label on GitHub — I had to define one. After testing a stricter version (requiring both CI *and* tests), which produced a heavily imbalanced dataset (~87% negative class), I settled on:

```
quality = 1 if (stars_per_month > median) AND (has_ci OR has_tests)
quality = 0 otherwise
```

A repo counts as "quality" if it's gaining stars faster than the median repo in the dataset **and** has at least one sign of engineering discipline (a CI pipeline or a tests folder). This is a deliberately simple, defensible proxy — not a claim about code quality itself, which isn't something you can fully infer from metadata.

## Dataset

- **2,218 repositories**, collected via the GitHub Search API across five topics: `machine-learning`, `deep-learning`, `nlp`, `computer-vision`, `data-science`
- For each repo: stars, forks, open issues, creation/push dates, README (full text + size), topic tags, and boolean flags for `has_readme`, `has_ci`, `has_tests`
- Class balance: ~71% not-quality / ~29% quality after combining two collection batches (the first single-topic batch had a more severe ~87/13 split)

## Features used

**Structured:** stars, forks, open issues, README size, repo age (days), days since last commit, has_readme

**Text (TF-IDF):**
- README content (500 features)
- Topic tags (100 features)

**Deliberately excluded:** `has_ci`, `has_tests`, and `stars_per_month` are *not* used as model features, since they were used to construct the label itself — including them would be data leakage (the model predicting from its own answer key).

## Results

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.51 | 0.66 | 0.57 |
| **Random Forest** | **0.89** | **0.80** | **0.81** | **0.80** |

Random Forest is reported as the primary model. Class imbalance was handled with `class_weight="balanced"` rather than resampling.

### Top predictive features
Repo activity signals (`stars`, `days_since_last_commit`, `forks`, `open_issues`) dominate, followed by documentation-related README vocabulary (`docs`, `documentation`, `contributing`, `install`).

## Known limitations

- **Possible indirect leakage via README badges.** Words like `workflows`, `actions`, `badge`, and `shields` rank among the top features — these likely come from CI-status badges embedded in README text. Even though `has_ci` was excluded as a direct feature, the model may be partially recovering that signal indirectly through badge-related vocabulary in the README.
- **Topic tags gave only a modest improvement** (F1 0.77 → 0.80) and didn't produce any single feature in the top 20 — likely because topic information overlaps with vocabulary already present in README text.
- **"Quality" is a proxy, not a ground truth.** Stars-per-month rewards popularity, which correlates with but doesn't equal code quality. A well-written internal tool with few stars would be scored "not quality" here.
- **Class 1 recall (0.81) means ~19% of true quality repos are still missed** — some real quality signal likely isn't captured by the current feature set.

## Project structure

```
reposcore/
├── app.py                          # Streamlit demo
├── notebooks/
│   ├── collect_repos.py            # Batch 1: search API collection
│   ├── enrich_repos.py             # Batch 1: README/CI/tests check
│   ├── fetch_readmes.py            # Batch 1: full README text
│   ├── collect_repos_batch2.py     # Batch 2: additional topics
│   ├── enrich_repos_batch2.py      # Batch 2: enrichment
│   ├── fetch_readmes_batch2.py     # Batch 2: README text
│   ├── fetch_topics.py             # Topic tags for full dataset
│   ├── build_dataset_v2.py         # Merge, clean, label
│   └── train_model.py              # Train, evaluate, save model
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone https://github.com/YOUR-USERNAME/reposcore.git
cd reposcore
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file with your own GitHub personal access token:
```
GITHUB_TOKEN=your_token_here
```

Run the demo (uses the pre-trained model):
```bash
streamlit run app.py
```

To retrain from scratch, run the scripts in `notebooks/` in order (collection → enrichment → README fetch → topics → dataset build → training).
