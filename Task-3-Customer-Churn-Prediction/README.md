# Task 3 — Customer Churn Prediction

A complete CodSoft machine-learning project for predicting whether a bank customer is likely to churn (`Exited`).

## Dataset
The project uses `data/Churn_Modelling.csv` (10,000 rows). Identifier/name columns are removed; categorical variables are one-hot encoded. The target is imbalanced (~20.4% churn).

## Model
A class-weighted **Random Forest** classifier is trained with a stratified 70/15/15 train/validation/test split. Evaluation includes accuracy, precision, recall, F1, ROC-AUC and PR-AUC.

## Run
```bash
pip install -r requirements.txt
python train.py
python predict.py --credit-score 650 --geography France --gender Female --age 40 --tenure 5 --balance 75000 --num-products 1 --has-card 1 --active 1 --salary 100000
streamlit run app.py
```

## Files
- `train.py` — train and save the model
- `predict.py` — command-line prediction
- `app.py` — Streamlit interface
- `models/churn_model.joblib` — trained pipeline
- `reports/` — evaluation metrics and feature importance
- `data/Churn_Modelling.csv` — supplied dataset
