# Candlestick Classifier — Streamlit App

This repository contains a Streamlit app that deploys a next-day candlestick direction classifier (XGBoost) trained on RELIANCE.NS daily OHLCV.

## Folder structure
- `streamlit_app/app.py` — main Streamlit application
- `streamlit_app/xgb_model.joblib` — trained model (copy from Colab)
- `streamlit_app/scaler.pkl` — trained StandardScaler (copy from Colab)
- `streamlit_app/best_model_meta.json` — feature order metadata
- `streamlit_app/model_stats.json` — test metrics
- `streamlit_app/sample_input.csv` — example CSV for testing

## Deploy (Streamlit Cloud)
1. Create a new GitHub repository and push this folder structure.
2. Ensure `xgb_model.joblib` and `scaler.pkl` are present in `streamlit_app/`.
3. On Streamlit Cloud, create a new app and point it to the repository and branch.
4. Set `Main file` to `streamlit_app.app`.
5. Add `requirements.txt` if Streamlit Cloud asks for dependencies.
6. Deploy. Check logs if any missing packages are reported.

## Notes
- The model is experimental and accuracy may be low for next-day predictions using daily candles. See the in-app disclaimer.
- For better performance, consider: predicting multi-day returns, intraday data, or adding macro/regime features.
