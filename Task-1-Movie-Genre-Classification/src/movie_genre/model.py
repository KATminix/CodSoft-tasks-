"""Model construction, metrics, and serialization helpers."""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline() -> Pipeline:
    """Create the TF-IDF text pipeline and linear SVM classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=100_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )


def evaluate(model: Pipeline, texts, labels) -> tuple[dict, str]:
    predictions = model.predict(texts)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "samples": int(len(labels)),
    }
    report = classification_report(labels, predictions, zero_division=0)
    return metrics, report


def save_evaluation(output_dir: Path, name: str, metrics: dict, report: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{name}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)
