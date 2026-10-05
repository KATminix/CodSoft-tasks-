"""Predict a movie genre from a plot summary or a text file of summaries."""

import argparse
from pathlib import Path

from src.movie_genre.model import load_model


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot", nargs="?", help="One plot summary to classify")
    parser.add_argument("--input-file", type=Path, help="UTF-8 file with one plot per line")
    parser.add_argument("--model", type=Path, default=ROOT / "models/movie_genre_classifier.joblib")
    args = parser.parse_args()
    if bool(args.plot) == bool(args.input_file):
        parser.error("provide exactly one plot argument or --input-file")
    return args


def main() -> None:
    args = parse_args()
    if args.input_file:
        plots = [line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        plots = [args.plot]
    if not plots:
        raise SystemExit("No plot summaries were found.")
    predictions = load_model(args.model).predict(plots)
    for index, prediction in enumerate(predictions, start=1):
        prefix = f"{index}. " if len(predictions) > 1 else ""
        print(f"{prefix}{prediction}")


if __name__ == "__main__":
    main()
