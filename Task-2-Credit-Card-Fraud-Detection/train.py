"""Train and evaluate the credit-card fraud detector."""
from pathlib import Path
import json
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from src.features import make_features, CATEGORICAL_COLUMNS

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "fraudTrain.csv"
TEST_FILE = ROOT / "data" / "fraudTest.csv"
MODEL_FILE = ROOT / "models" / "fraud_detector.joblib"
REPORT_DIR = ROOT / "reports"


def main():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError("Put fraudTrain.csv and fraudTest.csv in the data/ directory.")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X = make_features(train)
    Xt = make_features(test)
    y = train["is_fraud"].astype("int8")
    yt = test["is_fraud"].astype("int8")

    # Keep validation chronological to avoid training on future transactions.
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    params = dict(
        objective="binary", n_estimators=1000, learning_rate=0.05,
        num_leaves=63, max_depth=-1, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )

    # Refit on all historical training data using the selected iteration count.
    final_model = lgb.LGBMClassifier(**{**params, "n_estimators": model.best_iteration_})
    final_model.fit(X, y, categorical_feature=CATEGORICAL_COLUMNS,
                    callbacks=[lgb.log_evaluation(0)])

    probabilities = final_model.predict_proba(Xt)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")

    metrics = {
        "accuracy": accuracy_score(yt, predictions),
        "precision": precision_score(yt, predictions, zero_division=0),
        "recall": recall_score(yt, predictions, zero_division=0),
        "f1": f1_score(yt, predictions, zero_division=0),
        "roc_auc": roc_auc_score(yt, probabilities),
        "pr_auc": average_precision_score(yt, probabilities),
        "best_iteration": int(model.best_iteration_),
    }

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_FILE.parent.mkdir(exist_ok=True)
    categorical_levels = {
        c: list(X[c].cat.categories.astype(str)) for c in CATEGORICAL_COLUMNS
    }
    joblib.dump({
        "model": final_model,
        "categorical_levels": categorical_levels,
        "features": list(X.columns),
        "threshold": 0.5,
    }, MODEL_FILE, compress=3)

    report = classification_report(
        yt, predictions, target_names=["Legitimate", "Fraud"],
        output_dict=True, zero_division=0
    )
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({**metrics, "classification_report": report}, f, indent=2)

    cm = confusion_matrix(yt, predictions)
    pd.DataFrame(cm, index=["Actual Legitimate", "Actual Fraud"],
                 columns=["Pred Legitimate", "Pred Fraud"]).to_csv(REPORT_DIR / "confusion_matrix.csv")
    pd.DataFrame({"feature": final_model.feature_name_,
                  "importance": final_model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
