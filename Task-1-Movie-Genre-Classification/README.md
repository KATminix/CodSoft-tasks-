# Task 1: Movie Genre Classification

Movie genre classification from plot summaries using TF-IDF features and a
linear support vector classifier (`LinearSVC`). The data format is the IMDb
Genre Classification dataset used for CodSoft Task 1.

## Project layout

```text
Task-1-Movie-Genre-Classification/
├── app.py                  # Streamlit interface
├── predict.py              # Command-line prediction
├── train.py                # Training, validation, and test evaluation
├── requirements.txt
├── data/                   # CodSoft dataset files
├── metrics/                # Created by train.py
├── models/                 # Trained model is saved here
└── src/movie_genre/        # Dataset parsing and model helpers
```

## Dataset

The **Genre Classification Dataset IMDb** used for the CodSoft task is included
in `data/`. Its files are:

- `train_data.txt` — rows formatted as `id ::: title ::: genre ::: description`.
- `test_data.txt` — rows formatted as `id ::: title ::: description`.
- `test_data_solution.txt` — labeled test rows in the same format as the training data.
- `description.txt` — source file describing the dataset format.

The dataset is also available from
[Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
The labeled solution file is used for held-out test evaluation by `train.py`.

## Install

Python 3.10 or newer is recommended.

```bash
cd Task-1-Movie-Genre-Classification
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train, validate, and evaluate

```bash
python train.py
```

The script parses the task's `:::`-delimited text files, reserves a stratified
validation split, fits a TF-IDF + `LinearSVC` pipeline on the training portion,
and reports validation accuracy, macro/weighted F1, and a per-class
classification report. When `test_data_solution.txt` is available, it then
evaluates the held-out test examples and reports the same metrics. It refits
the final pipeline on all labeled training rows and saves it to
`models/movie_genre_classifier.joblib`; metrics and reports are written to
`metrics/` as JSON and text files.

Options are available for nonstandard dataset locations and validation split:

```bash
python train.py --data-dir path/to/data --validation-size 0.2 --random-state 42
```

## Predict

Train once first so the saved model exists, then pass a plot summary:

```bash
python predict.py "A detective investigates a series of mysterious murders in a city."
```

To predict from a UTF-8 text file (one plot per line):

```bash
python predict.py --input-file plots.txt
```

Use `--model path/to/model.joblib` to load a model from another location.

## Streamlit app

```bash
streamlit run app.py
```

Paste a plot summary into the app and select **Predict genre**. The app reads
the same saved pipeline produced by `train.py`.

## Notes

- The classifier predicts one genre label per movie, matching the task's
  single-label `LinearSVC` workflow.
- The complete pipeline is serialized, so inference applies the same TF-IDF
  vocabulary and preprocessing learned during training.
- The included dataset files are tracked by Git. Generated model and metric
  artifacts are ignored; run training locally to create them.

