import pandas as pd
import joblib
import streamlit as st

model =joblib.load("taj_model.pkl")
ct = joblib.load("deeno_encoder.pkl")

qual = st.text_input("enter you quqlification: ")
sector = st.text_input("enter your sector: ")
level = st.text_input ("enter your level: ")

if st.button("summit"):
    sample =pd.DataFrame({
        "qual": [qual],
        "sector": [sector],
        "level": [level]
})

    encoded = ct.transform(sample)
    prediction = model.predict(encoded)

    st.success(f"Salary prediction: {prediction}")
