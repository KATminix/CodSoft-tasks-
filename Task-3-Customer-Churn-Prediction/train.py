"""Train the customer churn classifier."""
from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'Churn_Modelling.csv'
MODEL=ROOT/'models'/'churn_model.joblib'
df=pd.read_csv(DATA)
y=df.pop('Exited')
X=df.drop(columns=['RowNumber','CustomerId','Surname'])
cat=['Geography','Gender']; num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
model=RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
pipe=Pipeline([('preprocess',pre),('classifier',model)])
pipe.fit(X,y); MODEL.parent.mkdir(exist_ok=True); joblib.dump(pipe,MODEL)
print(f'Saved model to {MODEL}')
