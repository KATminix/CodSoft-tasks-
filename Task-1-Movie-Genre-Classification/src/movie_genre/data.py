"""Read the CodSoft IMDb genre dataset's `:::`-delimited text files."""

from pathlib import Path

import pandas as pd


def _read_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(
        path,
        sep=r"\s*:::\s*",
        engine="python",
        header=None,
        names=columns,
        usecols=range(len(columns)),
        quoting=3,
        on_bad_lines="skip",
        encoding="utf-8",
    )
    for column in columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame


def load_training_data(path: Path) -> pd.DataFrame:
    """Load labeled training rows with id, title, genre, and plot fields."""
    rows = _read_rows(path, ["id", "title", "genre", "plot"])
    rows = rows[(rows["genre"] != "") & (rows["plot"] != "")]
    if rows.empty:
        raise ValueError(f"No usable labeled rows found in {path}")
    return rows


def load_test_data(path: Path) -> pd.DataFrame:
    """Load unlabeled test rows with id, title, and plot fields."""
    rows = _read_rows(path, ["id", "title", "plot"])
    return rows[rows["plot"] != ""].reset_index(drop=True)


def load_test_solution(path: Path) -> pd.DataFrame:
    """Load labeled test rows with id, title, genre, and plot fields."""
    rows = _read_rows(path, ["id", "title", "genre", "plot"])
    rows = rows[(rows["genre"] != "") & (rows["plot"] != "")]
    return rows.reset_index(drop=True)
