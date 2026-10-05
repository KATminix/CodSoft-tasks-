import streamlit as st, joblib, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'models'/'churn_model.joblib')
st.set_page_config(page_title='Customer Churn Predictor',page_icon='📉',layout='centered')
st.title('Customer Churn Predictor')
st.caption('Random Forest model trained on the CodSoft Churn Modelling dataset.')
with st.form('customer'):
 c1,c2=st.columns(2)
 with c1:
  credit=st.number_input('Credit Score',300,850,650); geo=st.selectbox('Geography',['France','Germany','Spain']); gender=st.selectbox('Gender',['Female','Male']); age=st.number_input('Age',18,100,40); tenure=st.number_input('Tenure (years)',0,10,5)
 with c2:
  balance=st.number_input('Balance',0.0,300000.0,75000.0); products=st.number_input('Number of Products',1,4,1); card=st.selectbox('Has Credit Card',[1,0]); active=st.selectbox('Is Active Member',[1,0]); salary=st.number_input('Estimated Salary',0.0,250000.0,100000.0)
 submit=st.form_submit_button('Predict Churn')
if submit:
 row=pd.DataFrame([{'CreditScore':credit,'Geography':geo,'Gender':gender,'Age':age,'Tenure':tenure,'Balance':balance,'NumOfProducts':products,'HasCrCard':card,'IsActiveMember':active,'EstimatedSalary':salary}])
 prob=model.predict_proba(row)[0,1]
 st.metric('Churn probability',f'{prob:.1%}')
 if prob>=.5: st.error('Higher churn risk')
 else: st.success('Lower churn risk')
