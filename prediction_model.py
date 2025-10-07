from pycaret.regression import models, setup, create_model, compare_models, finalize_model, plot_model, save_model, get_config, predict_model
from pycaret.datasets import get_data
from datetime import datetime, timedelta
import pandas as pd
import csv
import streamlit as st

@st.cache_data
def get_model():
    return create_model("lr")

def predict(df):
    df.to_csv("USD_JPY.csv", index=False, header = False, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
    df = pd.read_csv("USD_JPY.csv")
    df.columns = ["Datetime","Close","High","Low","Open","Volume"]
    df = df.drop("Datetime", axis=1)
    df = df.drop("Volume", axis=1)
    prediction_row = df.take([-1])
    df["next_close"] = df["Close"].copy()
    df["next_close"] = df["next_close"].shift(-1)
    df = df.iloc[:-1]
    df.reset_index(drop=True, inplace=True)

    exp = setup(data=df, target="next_close", session_id=123)

    model = get_model()
    s = exp.predict_model(model, data=prediction_row)
    return round(float(s.prediction_label), 3)