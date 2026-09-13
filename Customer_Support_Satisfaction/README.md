# Customer Support Ticket Analysis & Satisfaction Prediction

Explores a customer support ticket dataset and builds classification models to predict
`Customer Satisfaction Rating` (1–5) from ticket, customer, and product attributes — with
the goal of flagging tickets at risk of a poor rating before they close.

## Dataset

[Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
(Kaggle) — 8,469 support tickets with customer, product, ticket, and resolution details.
A copy is included at `data/customer_support_dataset.csv`.

## What's in this repo

```
.
├── data/
│   └── customer_support_dataset.csv   # the raw dataset
├── src/
│   └── customer_support_satisfaction.py   # full analysis + modeling pipeline
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — shape, dtypes, missing values, duplicates, age/response distributions
2. **Target selection** — `Customer Satisfaction Rating` is only populated for closed
   tickets (2,769 of 8,469 rows); the dataset is filtered to those rows, and the
   now-constant `Ticket Status` column is dropped
3. **NLP experiment** — a lexicon-based sentiment score is computed on the free-text
   `Resolution` field and tested as a feature (see Results below)
4. **Feature engineering & encoding** — one-hot encoding for low-cardinality categoricals,
   frequency-encoding for the 42-value `Product Purchased` column, numeric scaling
5. **Modeling** — Logistic Regression, Random Forest (tuned via `GridSearchCV`), and an
   `SGDClassifier` (gradient descent), plus a from-scratch batch gradient descent
   implementation of logistic regression for a binary version of the target
6. **Evaluation** — accuracy, precision, recall, F1 (weighted), MAE, and "within ±1"
   accuracy (since the rating is ordinal), plus 5-fold cross-validation to confirm
   the results are stable

## Results

| Model | Accuracy | F1 | MAE | Within ±1 |
|---|---|---|---|---|
| SGD (Gradient Descent) | 0.213 | 0.211 | 1.644 | 0.514 |
| Random Forest (tuned) | 0.193 | 0.190 | 1.551 | 0.540 |
| Logistic Regression | 0.181 | 0.177 | 1.579 | 0.507 |

**Honest finding:** with 5 balanced rating classes, random guessing scores ~20%. All three
models land within a couple of points of that baseline, confirmed stable via 5-fold
cross-validation (std ≈ 0.01–0.02) and unmoved by the sentiment feature (correlation with
rating ≈ 0.01 — the `Resolution` text turns out to be randomly-generated placeholder
sentences, not real agent notes). None of the available attributes carry meaningful
predictive signal about satisfaction in this dataset. The pipeline itself — leakage-aware
feature dropping, proper encoding, hyperparameter tuning, an honestly-tested NLP feature,
and cross-validated, ordinal-aware evaluation — is complete and reusable on a dataset where
the label does carry signal.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Run

```bash
python src/customer_support_satisfaction.py
```

Reads `data/customer_support_dataset.csv` automatically (path is resolved relative to the
script, so this works from any working directory) and prints all EDA/metrics output,
saving each chart as it's generated (charts display via `plt.show()` — swap in `plt.savefig(...)`
in the script if you'd rather write them to files in a headless environment).

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn.
