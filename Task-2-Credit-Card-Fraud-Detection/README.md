# Task 2 — Credit Card Fraud Detection

A complete machine-learning workflow for detecting fraudulent credit-card transactions using the **CodSoft Credit Card Fraud Detection dataset**.

## Project overview

The dataset contains transaction details such as amount, merchant, category, location, customer demographics and timestamps. Fraud is highly imbalanced, so the project evaluates the model with **precision, recall, F1, ROC-AUC and PR-AUC**, rather than relying on accuracy alone.

### Model

The final model is **LightGBM**, a gradient-boosted decision-tree classifier. The pipeline engineers time, age and geographic-distance features and treats selected identifiers/categories as categorical features.

### Engineered features

- Transaction amount and city population
- Customer and merchant latitude/longitude
- Transaction hour, weekday, month and day
- Approximate customer-to-merchant distance in km
- Customer age at transaction time
- Merchant, category, gender, state, job, city, ZIP and card-number categorical information

Raw names, street text and transaction IDs are not used as predictive features.

## Dataset

Place these files in `data/`:

```text
fraudTrain.csv
fraudTest.csv
```

The original dataset is not included in this repository because the CSV files are very large.

## Installation

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This performs a chronological validation split, selects the number of boosting rounds with early stopping, refits on all training data, evaluates on the supplied test set, and saves:

- `models/fraud_detector.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.csv`
- `reports/feature_importance.csv`

## Command-line prediction

Example:

```bash
python predict.py --amount 120.50 --category shopping_net --gender F --state NC
```

## Web app

```bash
streamlit run app.py
```

The Streamlit interface accepts transaction details and displays the estimated fraud probability.

## Test-set results

The included trained model was evaluated on the provided held-out `fraudTest.csv`:

| Metric | Score |
|---|---:|
| Accuracy | 99.84% |
| Precision | 89.56% |
| Recall | 66.81% |
| F1-score | 76.53% |
| ROC-AUC | 99.12% |
| PR-AUC | 79.52% |

The test set contains a much smaller fraud proportion than legitimate transactions, which is why accuracy alone is not an adequate measure of performance.

## Project structure

```text
Task-2-Credit-Card-Fraud-Detection/
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── fraudTrain.csv / fraudTest.csv  # local only
├── models/
│   └── fraud_detector.joblib
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   └── feature_importance.csv
└── src/
    └── features.py
```

## Important note

This is an educational CodSoft project. A fraud probability from this model should not be treated as a production financial decision without additional validation, monitoring, calibration, security controls and domain review.
