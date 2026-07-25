# ⭐ RepoScore: GitHub Repository Quality Predictor

RepoScore predicts whether a GitHub repository is likely to be a "high-quality" project based on its metadata, README content, and topic tags. Enter any public GitHub repo and get an instant quality prediction, powered by a Random Forest classifier trained on ~2,200 real repositories.

**Live demo:** [reposcoree.streamlit.app](https://reposcoree.streamlit.app/) — paste a repo (e.g. `scikit-learn/scikit-learn`) and get a prediction with confidence score and a feature-level explanation.

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

Earlier versions of this README reported 0.89 accuracy / 0.80 F1 from a single train/test split. That number was optimistic for two compounding reasons, both now fixed:

1. **Badge-markup leakage.** CI-status badges embedded in README text (`![CI](...shields.io...)`) let the model partially recover the `has_ci` signal indirectly, even though `has_ci` itself was correctly excluded as a direct feature.
2. **Single-split variance.** A lucky 80/20 split overstated how well the model generalizes on ~2,200 rows.

The honest numbers, from 5-fold stratified cross-validation on badge-stripped README text:

| Metric | Mean ± std (5-fold CV) |
|---|---|
| F1 (class 1) | 0.610 ± 0.053 |
| Precision (class 1) | 0.904 ± 0.048 |
| Recall (class 1) | 0.463 ± 0.057 |

For comparison, here's the effect of the badge-stripping fix in isolation, both evaluated the same way (5-fold CV):

| Setup | F1 | Precision | Recall |
|---|---|---|---|
| Raw README (badges included) | 0.658 ± 0.043 | 0.898 ± 0.027 | 0.520 ± 0.049 |
| **Badges stripped (current)** | **0.610 ± 0.053** | **0.904 ± 0.048** | **0.463 ± 0.057** |

Stripping badges drops F1 by about 5 points — that gap **is** the leaked signal being removed. Precision barely moves; recall drops, meaning some of what the raw-badge model was "detecting" was really just reading CI badges, not README quality.

A single held-out 80/20 split (used only to produce the confusion matrix and save the deployed model) gives:

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.50 | 0.65 | 0.56 | — |
| **Random Forest** | **0.83** | **0.88** | **0.50** | **0.63** | **0.94** |

Confusion matrix (Random Forest, held-out set, `[[TN, FP], [FN, TP]]`):
```
[[306   9]
 [ 65  64]]
```
Full metrics are saved to `models/metrics_report.json` on every training run, so results stay auditable instead of hand-copied into this README.

Class imbalance was handled with `class_weight="balanced"` rather than resampling.

### Top predictive features
After removing badge markup, repo activity signals (`stars`, `days_since_last_commit`, `forks`, `open_issues`) still dominate, followed by documentation-related README vocabulary (`docs`, `documentation`, `contributing`, `install`). The badge-related tokens (`workflows`, `shields`, `svg`, `logo`) that used to appear in the top 15 are gone.

## Known limitations

- **Recall is the real weak point (0.46 CV, 0.50 held-out).** After removing the badge-leakage shortcut, the model misses roughly half of true "quality" repos — it's a real limitation, not just a hidden strength, since it means the current feature set isn't fully capturing what makes a repo "quality" by this definition.
- **Topic tags gave only a modest improvement** and didn't produce any single feature in the top 20 — likely because topic information overlaps with vocabulary already present in README text.
- **"Quality" is a proxy, not a ground truth.** Stars-per-month rewards popularity, which correlates with but doesn't equal code quality. A well-written internal tool with few stars would be scored "not quality" here.
- **Badge stripping is regex-based, not exhaustive.** It targets shields.io, badge.fury.io, and similarly-structured badge hosts/markdown patterns; some CI-signal likely still leaks through badge formats the regex doesn't cover.
- **No temporal validation.** The dataset is a single point-in-time snapshot; there's no check for how well the model generalizes to repos created after collection, or to topics outside the five collected here.

## Explainability

The Streamlit app shows a per-prediction SHAP breakdown alongside the score — which specific features (README vocabulary, stars, activity, etc.) pushed this particular repo's prediction toward "high quality" or "low quality." This turns the Random Forest's output from a bare number into something a user can sanity-check against the repo they just looked up.

## Project structure

```
reposcore/
├── app.py                          # Streamlit demo (predicts + explains with SHAP)
├── reposcore_utils.py              # Shared preprocessing (badge stripping) used by
│                                    #   both training and inference, so they can't drift apart
├── notebooks/
│   ├── collect_repos.py            # Batch 1: search API collection
│   ├── enrich_repos.py             # Batch 1: README/CI/tests check
│   ├── fetch_readmes.py            # Batch 1: full README text
│   ├── collect_repos_batch2.py     # Batch 2: additional topics
│   ├── enrich_repos_batch2.py      # Batch 2: enrichment
│   ├── fetch_readmes_batch2.py     # Batch 2: README text
│   ├── fetch_topics.py             # Topic tags for full dataset
│   ├── build_dataset_v2.py         # Merge, clean, label
│   └── train_model.py              # Train, CV-evaluate, save model + metrics report
├── tests/
│   └── test_reposcore_utils.py     # Guards the badge-stripping fix against regressions
├── .github/workflows/ci.yml        # Runs tests on every push/PR
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # + pytest, for running the test suite
└── README.md
```

## Running locally

```bash
git clone https://github.com/mohit4236b-svg/reposcore.git
cd reposcore
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
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

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run automatically on every push/PR via GitHub Actions (see `.github/workflows/ci.yml`). They currently cover the badge-stripping preprocessing step — the piece of logic that fixes the leakage bug described below — since a silent regression there would let the label-leakage signal back in without anyone noticing.
