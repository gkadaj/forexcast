from pycaret.regression import models, setup, create_model, compare_models, finalize_model, plot_model, save_model, get_config, predict_model, load_model
from pycaret.datasets import get_data
from datetime import datetime, timedelta
import pandas as pd
import csv
import streamlit as st
from pathlib import Path

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


    # Check if the file exists
    if not Path("fx_model.pkl").exists():
        setup(data=df, target="next_close", session_id=123)
        model = compare_models()
        final_model = finalize_model(model)
        save_model(final_model, "fx_model")

    final_model = load_model("fx_model")
    s = predict_model(final_model, data=prediction_row)
    
    print(round(float(s.prediction_label), 3))
    return round(float(s.prediction_label), 3)