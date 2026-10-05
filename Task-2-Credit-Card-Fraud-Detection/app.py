import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from src.features import make_features

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "models" / "fraud_detector.joblib")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.caption("CodSoft Task 2 — LightGBM binary classifier")

with st.form("transaction"):
    amount = st.number_input("Transaction amount", min_value=0.0, value=50.0, step=1.0)
    category = st.selectbox("Category", ["misc_net", "grocery_pos", "shopping_net", "shopping_pos", "gas_transport", "food_dining", "personal_care", "home", "entertainment", "kids_pets", "health_fitness", "travel", "grocery_net", "misc_pos"])
    gender = st.selectbox("Gender", ["F", "M"])
    state = st.text_input("State", "NC")
    city = st.text_input("City", "Moravian Falls")
    job = st.text_input("Job", "Engineer")
    merchant = st.text_input("Merchant", "fraud_Rippin, Kub and Mann")
    cc_num = st.text_input("Card number", "2703186189652095")
    trans_date_trans_time = st.text_input("Transaction date/time", "2020-06-21 12:14:25")
    dob = st.text_input("Date of birth", "1988-03-09")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Customer latitude", value=36.0788)
        long = st.number_input("Customer longitude", value=-81.1781)
        zip_code = st.text_input("ZIP", "28654")
        city_pop = st.number_input("City population", min_value=0, value=3495)
    with col2:
        merch_lat = st.number_input("Merchant latitude", value=36.011293)
        merch_long = st.number_input("Merchant longitude", value=-82.048315)
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    row = {
        "amt": amount, "category": category, "gender": gender, "state": state,
        "city": city, "job": job, "merchant": merchant, "cc_num": cc_num,
        "trans_date_trans_time": trans_date_trans_time, "dob": dob,
        "lat": lat, "long": long, "zip": zip_code, "city_pop": city_pop,
        "merch_lat": merch_lat, "merch_long": merch_long,
    }
    X = make_features(pd.DataFrame([row]), bundle["categorical_levels"])
    probability = float(bundle["model"].predict_proba(X)[:, 1][0])
    if probability >= bundle["threshold"]:
        st.error(f"⚠️ Potential fraud — probability {probability:.2%}")
    else:
        st.success(f"✓ Likely legitimate — fraud probability {probability:.2%}")
    st.progress(min(probability, 1.0), text="Fraud probability")
