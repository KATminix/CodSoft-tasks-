"""Predict churn for one customer from command-line arguments."""
from pathlib import Path
import argparse, joblib, pandas as pd
ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'models'/'churn_model.joblib')
p=argparse.ArgumentParser()
p.add_argument('--credit-score',type=int,required=True); p.add_argument('--geography',required=True); p.add_argument('--gender',required=True)
p.add_argument('--age',type=int,required=True); p.add_argument('--tenure',type=int,required=True); p.add_argument('--balance',type=float,required=True)
p.add_argument('--num-products',type=int,required=True); p.add_argument('--has-card',type=int,choices=[0,1],required=True); p.add_argument('--active',type=int,choices=[0,1],required=True); p.add_argument('--salary',type=float,required=True)
a=p.parse_args(); row=pd.DataFrame([{'CreditScore':a.credit_score,'Geography':a.geography,'Gender':a.gender,'Age':a.age,'Tenure':a.tenure,'Balance':a.balance,'NumOfProducts':a.num_products,'HasCrCard':a.has_card,'IsActiveMember':a.active,'EstimatedSalary':a.salary}])
prob=model.predict_proba(row)[0,1]; print(f'Churn probability: {prob:.2%}'); print('Prediction:', 'LIKELY TO CHURN' if prob>=.5 else 'LIKELY TO STAY')
