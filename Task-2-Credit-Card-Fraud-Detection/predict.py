"""Predict whether one transaction is fraudulent."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from src.features import make_features

ROOT = Path(__file__).resolve().parent
BUNDLE = joblib.load(ROOT / "models" / "fraud_detector.joblib")


def predict(row):
    df = pd.DataFrame([row])
    X = make_features(df, BUNDLE["categorical_levels"])
    probability = float(BUNDLE["model"].predict_proba(X)[:, 1][0])
    return probability, probability >= BUNDLE["threshold"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--amount", type=float, required=True)
    p.add_argument("--category", default="misc_net")
    p.add_argument("--gender", default="F")
    p.add_argument("--state", default="NC")
    p.add_argument("--city", default="Moravian Falls")
    p.add_argument("--job", default="Psychologist, counselling")
    p.add_argument("--cc-num", dest="cc_num", default="2703186189652095")
    p.add_argument("--merchant", default="fraud_Rippin, Kub and Mann")
    p.add_argument("--zip", default="28654")
    p.add_argument("--date", default="2020-06-21 12:14:25")
    p.add_argument("--dob", default="1988-03-09")
    p.add_argument("--lat", type=float, default=36.0788)
    p.add_argument("--long", type=float, default=-81.1781)
    p.add_argument("--city-pop", dest="city_pop", type=int, default=3495)
    p.add_argument("--merch-lat", dest="merch_lat", type=float, default=36.011293)
    p.add_argument("--merch-long", dest="merch_long", type=float, default=-82.048315)
    args = vars(p.parse_args())
    args["trans_date_trans_time"] = args.pop("date")
    probability, fraud = predict(args)
    print(f"Fraud probability: {probability:.2%}")
    print("Prediction: FRAUD" if fraud else "Prediction: LEGITIMATE")
