# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import ta
import shutil
import subprocess
import sys

# ------------------------------------------------------------
#                    DOWNLOAD MODEL (Google Drive)
# ------------------------------------------------------------
# Replace with your Google Drive file id (you provided it)
GDRIVE_ID = "1lUxvKR8ISLQNt0ws8p5nnBYnFqmgzcjP"
MODEL_FILENAME = "xgb_model.joblib"
MODEL_DEST_PATH = os.path.join(os.getcwd(), "streamlit_app", MODEL_FILENAME)

def ensure_gdown_installed():
    try:
        import gdown  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        # re-import after install
        import gdown  # noqa: F401

def download_model_from_gdrive(gdrive_id, dest_path):
    """
    Download file from Google Drive (uc?id=) using gdown.
    The function downloads to a temp location first then moves into dest_path.
    """
    ensure_gdown_installed()
    import gdown

    # temp path
    tmp_path = f"/tmp/{MODEL_FILENAME}"
    # skip download if already in dest
    if os.path.exists(dest_path):
        return dest_path

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    st.info("Model not found locally. Downloading model from Google Drive (one-time)...")
    try:
        # download to /tmp
        downloaded = gdown.download(url, tmp_path, quiet=False)
        if downloaded is None:
            raise RuntimeError("gdown failed to download the model.")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # move to final destination
        shutil.move(tmp_path, dest_path)
        st.success("Model downloaded and saved.")
        return dest_path
    except Exception as e:
        # if download failed, show error and continue (app will stop later if model missing)
        st.error(f"Model download failed: {e}")
        raise

# Attempt download if missing
try:
    if not os.path.exists(MODEL_DEST_PATH):
        download_model_from_gdrive(GDRIVE_ID, MODEL_DEST_PATH)
except Exception:
    # allow load_artifacts to show helpful error; stop only if necessary below
    pass

# ------------------------------------------------------------
#                    STREAMLIT CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="Candlestick Classifier", layout="wide")

# file names used by rest of the app (must match downloaded model name)
MODEL_FILE = os.path.join("streamlit_app", "xgb_model.joblib")
SCALER_FILE = os.path.join("streamlit_app", "scaler.pkl")
STATS_FILE = os.path.join("streamlit_app", "model_stats.json")
META_FILE = os.path.join("streamlit_app", "best_model_meta.json")


# ------------------------------------------------------------
#                    LOAD ARTIFACTS
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}

    if os.path.exists(MODEL_FILE):
        artifacts["model"] = joblib.load(MODEL_FILE)
    else:
        artifacts["model"] = None

    if os.path.exists(SCALER_FILE):
        artifacts["scaler"] = joblib.load(SCALER_FILE)
    else:
        artifacts["scaler"] = None

    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                artifacts["stats"] = json.load(f)
        except:
            artifacts["stats"] = None
    else:
        artifacts["stats"] = None

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r") as f:
                artifacts["meta"] = json.load(f)
        except:
            artifacts["meta"] = None
    else:
        artifacts["meta"] = None

    return artifacts


art = load_artifacts()

# fail-safe checks
if art["model"] is None:
    st.error("❌ Model file not found. Please ensure xgb_model.joblib is present under streamlit_app/ or that the Google Drive link is correct.")
    st.stop()

if art["scaler"] is None:
    st.error("❌ Scaler file missing. Upload scaler.pkl inside streamlit_app/.")
    st.stop()

if art["meta"] is None or "feature_columns" not in art["meta"]:
    st.error("❌ Metadata missing or invalid. Please check best_model_meta.json.")
    st.stop()

FEATURE_COLS = art["meta"]["feature_columns"]


# ------------------------------------------------------------
#              FEATURE ENGINEERING PIPELINE
# ------------------------------------------------------------
def compute_features_from_ohlcv(df_raw):
    df = df_raw.copy().reset_index(drop=True)

    df["return_1"] = df["Close"].pct_change().fillna(0)
    df["body"] = df["Close"] - df["Open"]
    df["body_abs"] = df["body"].abs()

    co_max = np.maximum(df["Close"].values, df["Open"].values)
    co_min = np.minimum(df["Close"].values, df["Open"].values)

    df["upper_wick"] = df["High"].values - co_max
    df["lower_wick"] = co_min - df["Low"].values
    df["range"] = df["High"] - df["Low"]

    WINDOWS = [3, 5, 10, 20]
    for w in WINDOWS:
        df[f"vol_mean_{w}"] = df["Volume"].rolling(w).mean().bfill()
        df[f"close_mean_{w}"] = df["Close"].rolling(w).mean().bfill()
        df[f"close_std_{w}"] = df["Close"].rolling(w).std().fillna(0)

    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14, fillna=True).rsi()

    macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14, fillna=True
    ).average_true_range()
    df["ema_12"] = ta.trend.EMAIndicator(df["Close"], window=12, fillna=True).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(df["Close"], window=26, fillna=True).ema_indicator()

    df["vol_chg"] = df["Volume"].pct_change().fillna(0)
    df["vol_rel"] = df["Volume"] / df["Volume"].rolling(20).mean().fillna(1)

    LAGS = [1, 2, 3]
    for lag in LAGS:
        df[f"return_1_lag{lag}"] = df["return_1"].shift(lag)
        df[f"body_lag{lag}"] = df["body"].shift(lag)
        df[f"upper_wick_lag{lag}"] = df["upper_wick"].shift(lag)
        df[f"lower_wick_lag{lag}"] = df["lower_wick"].shift(lag)
        df[f"rsi_14_lag{lag}"] = df["rsi_14"].shift(lag)
        df[f"macd_lag{lag}"] = df["macd"].shift(lag)
        df[f"atr_14_lag{lag}"] = df["atr_14"].shift(lag)
        df[f"ema_12_lag{lag}"] = df["ema_12"].shift(lag)
        df[f"ema_26_lag{lag}"] = df["ema_26"].shift(lag)
        df[f"vol_rel_lag{lag}"] = df["vol_rel"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


# ------------------------------------------------------------
#              PREDICTION ENGINE
# ------------------------------------------------------------
def predict_from_ohlcv(df_input, threshold=0.5):
    df_feat = compute_features_from_ohlcv(df_input)

    if df_feat.shape[0] == 0:
        raise ValueError("Not enough rows after feature engineering.")

    last_row = df_feat.iloc[[-1]]

    missing = [c for c in FEATURE_COLS if c not in last_row.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = last_row[FEATURE_COLS].values
    Xs = art["scaler"].transform(X)

    model = art["model"]
    prob = float(model.predict_proba(Xs)[0][1])
    pred = 1 if prob >= threshold else 0

    return {
        "prob": prob,
        "pred": pred,
        "features": last_row[FEATURE_COLS].iloc[0].to_dict()
    }


# ------------------------------------------------------------
#                       UI LAYOUT
# ------------------------------------------------------------
st.title("📊 Candlestick Classifier — Next-Day Direction")

with st.sidebar:
    st.header("Input Settings")

    mode = st.radio("Select Input Method:", ("Fetch RELIANCE.NS", "Upload CSV (OHLCV)"))
    days = st.number_input("Days of history", min_value=30, max_value=400, value=60)
    threshold = st.slider("Decision Threshold (Bullish)", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")
    if art["stats"]:
        acc = art["stats"]["metrics"].get("accuracy")
        if acc:
            st.metric("Model Accuracy", f"{acc*100:.2f}%")

# ------------------------------------------------------------
#                INPUT HANDLING
# ------------------------------------------------------------
df_input = None

if mode == "Fetch RELIANCE.NS":
    st.info(f"Fetching last {days} days of RELIANCE.NS...")
    df = yf.download("RELIANCE.NS", period=f"{days}d", progress=False)

    if df.empty:
        st.error("Could not fetch data.")
        st.stop()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df_input = df[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)

else:
    uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        cols = {c.lower(): c for c in df.columns}

        required = ["open", "high", "low", "close", "volume"]
        if not all(r in cols for r in required):
            st.error("CSV must contain Open, High, Low, Close, Volume columns.")
            st.stop()

        df_input = df.rename(columns={cols["open"]: "Open",
                                      cols["high"]: "High",
                                      cols["low"]: "Low",
                                      cols["close"]: "Close",
                                      cols["volume"]: "Volume"})[
                   ["Open", "High", "Low", "Close", "Volume"]
               ].reset_index(drop=True)

if df_input is None:
    st.warning("Awaiting input…")
    st.stop()

# ------------------------------------------------------------
#            DISPLAY INPUT DATA
# ------------------------------------------------------------
st.subheader("📘 Input OHLCV Data")
st.dataframe(df_input.tail(50), height=300)

# ------------------------------------------------------------
#       CANDLESTICK VISUAL (LARGER)
# ------------------------------------------------------------
st.subheader("📈 Candlestick Pattern (Most Recent)")

plot_df = df_input.copy()
plot_df.index = pd.date_range(end=pd.Timestamp.today(), periods=len(plot_df))

fig, ax = plt.subplots(figsize=(11, 4))
mpf.plot(plot_df, type="candle", style="charles", ax=ax, volume=False)
st.pyplot(fig)

# ------------------------------------------------------------
#                RUN PREDICTION
# ------------------------------------------------------------
try:
    result = predict_from_ohlcv(df_input, threshold=threshold)
    prob = result["prob"]
    pred = result["pred"]
    label = "Bullish" if pred == 1 else "Bearish"

    st.markdown("## 🔮 Prediction Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Next-Day Trend", label, f"{prob*100:.1f}%")

    with col2:
        st.write(f"**Bullish Probability:** {prob:.4f}")
        if art["stats"]:
            st.write(f"**Model Test Accuracy:** {art['stats']['metrics']['accuracy']*100:.2f}%")

    st.subheader("🔍 Feature Values Used (Top 20)")
    feat_df = pd.DataFrame(
        list(result["features"].items()), columns=["Feature", "Value"]
    ).sort_values("Feature")
    st.dataframe(feat_df.head(20))

    st.subheader("🏆 Model Feature Importances (Top 20)")
    try:
        importances = art['model'].feature_importances_
        fi = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:20]
        st.table(pd.DataFrame(fi, columns=["Feature", "Importance"]))
    except:
        st.info("Feature importances unavailable for this model.")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ------------------------------------------------------------
#                       DISCLAIMER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("""
### ⚠️ **Disclaimer**
This model is experimental and based solely on historical OHLCV patterns.  
Daily next-day stock direction is highly noisy and difficult to predict reliably.  
This tool is intended for educational and demonstration purposes only.
""")
