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
import subprocess
import sys
import re
from io import BytesIO

# ------------------------------------------------------------
#                    CONFIG & PATHS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "xgb_model.joblib")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
STATS_FILE = os.path.join(BASE_DIR, "model_stats.json")
META_FILE = os.path.join(BASE_DIR, "best_model_meta.json")

# Google Drive model ID (if you want auto-download fallback)
GDRIVE_ID = "1lUxvKR8ISLQNt0ws8p5nnBYnFqmgzcjP"
MODEL_FILENAME = "xgb_model.joblib"

# ------------------------------------------------------------
#                    DOWNLOAD MODEL (Google Drive)
# ------------------------------------------------------------
def ensure_gdown_installed():
    try:
        import gdown  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown  # noqa: F401

def download_model_from_gdrive(gdrive_id, dest_path):
    """
    Download file from Google Drive (uc?id=) using gdown.
    Download directly into the app folder so paths remain consistent.
    """
    ensure_gdown_installed()
    import gdown

    if os.path.exists(dest_path):
        return dest_path

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    st.info("Model not found locally. Downloading model from Google Drive (one-time)...")
    try:
        downloaded = gdown.download(url, dest_path, quiet=False)
        if downloaded is None:
            raise RuntimeError("gdown failed to download the model. Check the Drive ID or sharing settings.")
        st.success("Model downloaded and saved.")
        return dest_path
    except Exception as e:
        st.error(f"Model download failed: {e}")
        raise

try:
    if not os.path.exists(MODEL_FILE):
        download_model_from_gdrive(GDRIVE_ID, MODEL_FILE)
except Exception:
    # allow load_artifacts to show helpful error later
    pass

# ------------------------------------------------------------
#                    STREAMLIT CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="Candlestick Classifier", layout="wide")

# ------------------------------------------------------------
#                    LOAD ARTIFACTS
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}

    if os.path.exists(MODEL_FILE):
        try:
            artifacts["model"] = joblib.load(MODEL_FILE)
        except Exception as e:
            artifacts["model"] = None
            st.error(f"Failed to load model file: {e}")
    else:
        artifacts["model"] = None

    if os.path.exists(SCALER_FILE):
        try:
            artifacts["scaler"] = joblib.load(SCALER_FILE)
        except Exception:
            artifacts["scaler"] = None
    else:
        artifacts["scaler"] = None

    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                artifacts["stats"] = json.load(f)
        except Exception:
            artifacts["stats"] = None
    else:
        artifacts["stats"] = None

    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r") as f:
                artifacts["meta"] = json.load(f)
        except Exception:
            artifacts["meta"] = None
    else:
        artifacts["meta"] = None

    return artifacts

art = load_artifacts()

# fail-safe checks
if art["model"] is None:
    st.error("❌ Model file not found or failed to load. Please ensure xgb_model.joblib is present under streamlit_app/ or that the Google Drive link is correct.")
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

# ----------- Sidebar Input Settings -----------
with st.sidebar:
    st.header("Input Settings")

    mode = st.radio(
        "Select Input Method:",
        ("Fetch RELIANCE.NS", "Upload CSV (OHLCV)", "Upload Image (OHLCV screenshot)")
    )
    days = st.number_input("Days of history", min_value=30, max_value=400, value=60)
    threshold = st.slider("Decision Threshold (Bullish)", 0.0, 1.0, 0.5, 0.01)

    st.markdown("---")
    if art["stats"]:
        acc = art["stats"]["metrics"].get("accuracy")
        if acc:
            st.metric("Model Accuracy", f"{acc*100:.2f}%")

# ----------------- INPUT HANDLING (CSV / IMAGE / YFINANCE) -----------------
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

elif mode == "Upload CSV (OHLCV)":
    uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

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

else:  # Image mode: allow OCR if available; require manual ticker as reliable fallback
    st.info("Upload an image (screenshot of OHLCV table or screenshot containing ticker). You may also enter the ticker below for a reliable fetch.")
    uploaded_img = st.file_uploader("Upload image (png/jpg)", type=["png", "jpg", "jpeg"])
    manual_ticker = st.text_input("Or enter ticker (e.g. RELIANCE.NS)", value="")

    image_bytes = None
    if uploaded_img is not None:
        image_bytes = uploaded_img.read()

    # try to import OCR libs
    ocr_available = True
    try:
        from PIL import Image
        import pytesseract
        import cv2
        import numpy as np
    except Exception:
        ocr_available = False

    parsed_df = None
    ticker_from_image = None

    # If OCR is available and the user uploaded an image, attempt OCR parsing (best-effort)
    if ocr_available and image_bytes is not None:
        try:
            pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1].copy()  # RGB->BGR

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_text = pytesseract.image_to_string(th)

            # First: attempt to find a table-like CSV in OCR text
            lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
            text_blob = "\n".join(lines)
            try:
                candidate = pd.read_csv(BytesIO(text_blob.encode()), engine="python")
                cols_lower = {c.lower(): c for c in candidate.columns}
                if all(k in cols_lower for k in ["open", "high", "low", "close", "volume"]):
                    parsed_df = candidate.rename(columns={cols_lower["open"]: "Open",
                                                          cols_lower["high"]: "High",
                                                          cols_lower["low"]: "Low",
                                                          cols_lower["close"]: "Close",
                                                          cols_lower["volume"]: "Volume"})[
                                 ["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            except Exception:
                parsed_df = None

            # If no valid parsed_df, try heuristic table parse from lines
            if parsed_df is None:
                header_idx = None
                for i, ln in enumerate(lines[:8]):  # header likely in first few lines
                    if re.search(r"open", ln, re.I) and re.search(r"high", ln, re.I) and re.search(r"low", ln, re.I):
                        header_idx = i
                        break
                if header_idx is not None:
                    header = re.sub(r"[^A-Za-z0-9,\t ]", "", lines[header_idx])
                    header_cols = re.split(r"[\t,]+|\s{2,}", header)
                    data_rows = []
                    for ln in lines[header_idx+1:]:
                        ln_clean = re.sub(r"[^\d\.\- ,\t]", " ", ln)
                        parts = re.split(r"[\t,]+|\s{2,}", ln_clean.strip())
                        if len(parts) >= len(header_cols):
                            data_rows.append(parts[:len(header_cols)])
                    if data_rows:
                        try:
                            df_candidate = pd.DataFrame(data_rows, columns=header_cols)
                            cols_lower = {c.lower(): c for c in df_candidate.columns}
                            if all(k in cols_lower for k in ["open", "high", "low", "close", "volume"]):
                                parsed_df = df_candidate.rename(columns={cols_lower["open"]: "Open",
                                                                         cols_lower["high"]: "High",
                                                                         cols_lower["low"]: "Low",
                                                                         cols_lower["close"]: "Close",
                                                                         cols_lower["volume"]: "Volume"})[
                                                ["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
                        except Exception:
                            parsed_df = None

            # If still not parsed, try to detect a ticker symbol in OCR text (e.g., RELIANCE.NS or TICKER)
            if parsed_df is None:
                ticker_matches = re.findall(r"\b[A-Z]{2,10}(?:\.NS|\.BO|\.NSX|\.BSE)?\b", ocr_text)
                if ticker_matches:
                    ticker_from_image = ticker_matches[0]
        except pytesseract.pytesseract.TesseractNotFoundError:
            ocr_available = False
        except Exception:
            parsed_df = None

    # Priority: 1) parsed_df from OCR  2) manual_ticker if provided  3) ticker_from_image from OCR
    if parsed_df is not None and not parsed_df.empty:
        df_input = parsed_df
    else:
        # prefer manual ticker (explicit user input)
        chosen_ticker = manual_ticker.strip().upper() if manual_ticker.strip() else (ticker_from_image or "")
        if chosen_ticker:
            st.info(f"Using ticker: {chosen_ticker}. Fetching last {days} days via yfinance...")
            try:
                df = yf.download(chosen_ticker, period=f"{days}d", progress=False)
                if df is None or df.empty:
                    st.error(f"Could not fetch data for {chosen_ticker}. Please verify the ticker or upload CSV.")
                    st.stop()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df_input = df[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to fetch ticker {chosen_ticker}: {e}")
                st.stop()
        else:
            # No parsed table, no ticker input/detection
            if not ocr_available:
                st.error("OCR libraries or Tesseract binary not available on this server. Please enter ticker manually in the text box above or upload a CSV.")
            else:
                st.error("Could not parse a valid OHLCV table from the image. Best practice: upload a high-resolution table screenshot with clear headers, or enter the ticker in the text box above or upload CSV.")
            st.stop()

# if still no df_input
if df_input is None:
    st.warning("Awaiting valid input…")
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
    except Exception:
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
