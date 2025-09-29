import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from keras.models import load_model
import numpy as np
import joblib
import pickle
import os

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Tesla Stock Mini-Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- HELPERS -----------------
def load_sklearn_model(path):
    """Try joblib.load first, then pickle.load."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def safe_last_close_from_df(df):
    if "Close" not in df.columns:
        raise KeyError("Uploaded CSV must contain a 'Close' column.")
    series = df["Close"].dropna().astype(float)
    if len(series) == 0:
        raise ValueError("No valid 'Close' values found in uploaded CSV.")
    return float(series.values[-1])

# ----------------- HEADER -----------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(120deg, #f6f9fc, #e9f5ff);
    }
    .title {
        text-align: center;
        color: #0A2647;
        font-size: 42px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>📊 Tesla Stock Mini-Analyser</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Dashboard • Moving Averages • Volatility • Predictions</div>", unsafe_allow_html=True)
st.write("---")

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Tesla_Motors.svg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Dashboard", "📂 Data", "📉 Analysis", "🤖 Prediction", "📊 Model Comparison", "ℹ️ About"]
)

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    path = "Tesla_Stock_2010_2024.csv"
    if not os.path.exists(path):
        st.error(f"Base data file not found: {path}. Upload it to the app folder.")
        return pd.DataFrame()
    df_local = pd.read_csv(path)
    if "Date" in df_local.columns:
        df_local["Date"] = pd.to_datetime(df_local["Date"])
    return df_local

df = load_data()

# ----------------- PAGES -----------------
if page == "📊 Dashboard":
    st.subheader("📊 Tesla Stock Dashboard")

    if df.empty:
        st.warning("Base dataset not loaded. Put `Tesla_Stock_2010_2024.csv` in the app directory.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Closing Price", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("Highest Price", f"${df['High'].max():.2f}")
        col3.metric("Lowest Price", f"${df['Low'].min():.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price", line=dict(color="blue")))
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Quick Summary")
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"))

elif page == "📂 Data":
    st.subheader("Raw Tesla Stock Data (2010–2024)")
    if df.empty:
        st.warning("Base dataset not loaded.")
    else:
        st.dataframe(df.tail(20), use_container_width=True)

elif page == "📉 Analysis":
    st.subheader("Stock Trend Analysis")
    if df.empty:
        st.warning("Base dataset not loaded.")
    else:
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["Volatility"] = df["Close"].rolling(20).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Price", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("20-day Volatility", f"{df['Volatility'].iloc[-1]:.2f}")
        col3.metric("50-day MA", f"${df['MA50'].iloc[-1]:.2f}")

elif page == "🤖 Prediction":
    st.subheader("📈 Short-Term Stock Price Prediction")

    uploaded_file = st.file_uploader("Upload Tesla Stock CSV for Prediction (optional)", type=["csv"])

    if uploaded_file:
        try:
            user_df = pd.read_csv(uploaded_file)
            if "Date" in user_df.columns:
                user_df["Date"] = pd.to_datetime(user_df["Date"])
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            user_df = pd.DataFrame()
    else:
        if not df.empty:
            st.info("No file uploaded — using internal dataset for prediction.")
            user_df = df.copy()
        else:
            st.warning("No uploaded file and base dataset missing. Upload a CSV with at least a 'Close' column.")
            user_df = pd.DataFrame()

    if user_df.empty:
        st.stop()

    st.write("Uploaded / Used Data Preview:")
    st.dataframe(user_df.tail())

    model_choice = st.selectbox(
        "Select Model for Prediction",
        ["LSTM", "Linear Regression", "Random Forest", "ARIMA", "XGBoost"]
    )

    try:
        last_close = safe_last_close_from_df(user_df)

        if model_choice == "LSTM":
            keras_model = load_model("tesla_stock_model.h5")
            scaler = joblib.load("tesla_scaler.pkl")
            last_price_scaled = scaler.transform([[last_close]])
            X_input = last_price_scaled.reshape(1, 1, 1)
            pred_scaled = keras_model.predict(X_input, verbose=0)
            pred_price = float(scaler.inverse_transform(pred_scaled)[0][0])

        elif model_choice in ["Linear Regression", "Random Forest", "XGBoost"]:
            model_file = f"{model_choice.lower().replace(' ', '_')}.pkl"
            skl_model = load_sklearn_model(model_file)
            pred_price = float(skl_model.predict([[last_close]])[0])

        elif model_choice == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA

            series = user_df["Close"].dropna().astype(float)
            n_points = len(series)

            if n_points < 30:
                st.warning(f"Series too short for ARIMA (length={n_points}). Using last close instead.")
                pred_price = float(series.values[-1])
            else:
                fit_series = series.iloc[-300:]  # last 300 points max
                with st.spinner("Fitting ARIMA model..."):
                    try:
                        arima_model = ARIMA(fit_series, order=(5, 1, 0))
                        arima_fit = arima_model.fit()
                        forecast = arima_fit.forecast(steps=1)
                        pred_price = float(forecast.values[0])
                    except Exception as e_arima:
                        st.error(f"⚠️ ARIMA fitting/forecast failed: {e_arima}. Falling back to last-close prediction.")
                        pred_price = float(series.values[-1])

        st.success(f"✅ Predicted Next-Day Closing Price using {model_choice}: **${pred_price:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

elif page == "📊 Model Comparison":
    st.subheader("📊 Model Performance Comparison")

    try:
        if not os.path.exists("metrics.pkl"):
            st.warning("metrics.pkl not found. Make sure you saved metrics from training to 'metrics.pkl'.")
        else:
            with open("metrics.pkl", "rb") as f:
                metrics_list = pickle.load(f)
            metrics_df = pd.DataFrame(metrics_list)
            st.dataframe(metrics_df)

            fig = go.Figure([go.Bar(x=metrics_df["Model"], y=metrics_df["RMSE"], marker=dict(color="skyblue"))])
            fig.update_layout(title="RMSE Comparison", xaxis_title="Model", yaxis_title="RMSE")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure([go.Bar(x=metrics_df["Model"], y=metrics_df["MAPE"], marker=dict(color="orange"))])
            fig2.update_layout(title="MAPE Comparison", xaxis_title="Model", yaxis_title="MAPE (%)")
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Unable to load metrics: {e}")

elif page == "ℹ️ About":
    st.subheader("About this Project")
    st.markdown(
        """
        This mini-analyser was built to:
        - Compute **Moving Averages & Volatility**
        - Provide **Short-term Predictions** using Multiple Models (LSTM, LR, RF, ARIMA, XGBoost)
        - Display **Interactive Stock Charts**
        - Summarize **Stock Performance with a Dashboard**
        - Compare **Model Performance Metrics**
        
        **Made with ❤️ using Streamlit**
        """
    )

# ----------------- FOOTER -----------------
st.markdown("<br><br><center>© 2025 Tesla Stock Mini-Analyser | Built with ❤️ Streamlit</center>", unsafe_allow_html=True)

