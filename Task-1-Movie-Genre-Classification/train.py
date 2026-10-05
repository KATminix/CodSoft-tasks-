"""Train, validate, and (when labels are available) evaluate the classifier."""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.movie_genre.data import load_test_solution, load_training_data
from src.movie_genre.model import build_pipeline, evaluate, save_evaluation, save_model


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/movie_genre_classifier.joblib")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise SystemExit("--validation-size must be between 0 and 1")

    train_rows = load_training_data(args.data_dir / "train_data.txt")
    class_counts = train_rows["genre"].value_counts()
    stratify = train_rows["genre"] if class_counts.min() >= 2 else None
    if stratify is None:
        print("Some genres have fewer than two examples; using an unstratified validation split.")

    x_train, x_val, y_train, y_val = train_test_split(
        train_rows["plot"],
        train_rows["genre"],
        test_size=args.validation_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    validation_model = build_pipeline()
    validation_model.set_params(classifier__random_state=args.random_state)
    validation_model.fit(x_train, y_train)
    validation_metrics, validation_report = evaluate(validation_model, x_val, y_val)
    print("Validation metrics:", validation_metrics)
    print("Validation classification report:\n", validation_report)
    save_evaluation(args.metrics_dir, "validation", validation_metrics, validation_report)

    solution_path = args.data_dir / "test_data_solution.txt"
    if solution_path.is_file():
        test_rows = load_test_solution(solution_path)
        test_model = build_pipeline()
        test_model.set_params(classifier__random_state=args.random_state)
        test_model.fit(train_rows["plot"], train_rows["genre"])
        test_metrics, test_report = evaluate(test_model, test_rows["plot"], test_rows["genre"])
        print("Test metrics:", test_metrics)
        print("Test classification report:\n", test_report)
        save_evaluation(args.metrics_dir, "test", test_metrics, test_report)
    else:
        print(
            f"No labeled test solution found at {solution_path}; "
            "held-out test metrics were skipped."
        )

    final_model = build_pipeline()
    final_model.set_params(classifier__random_state=args.random_state)
    final_model.fit(train_rows["plot"], train_rows["genre"])
    save_model(final_model, args.model_path)
    print(f"Saved final model to {args.model_path}")


if __name__ == "__main__":
    main()
