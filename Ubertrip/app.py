# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# xgboost optional import (we'll try/except to give helpful error if missing)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ---- PAGE CONFIG & STYLE ----
st.set_page_config(page_title="Uber Trips Forecast — Clean UI", layout="wide")
sns.set_style("whitegrid")

# small CSS for light, clean cards
st.markdown("""
    <style>
    .card { padding: 12px; border-radius: 8px; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .metric-row { display:flex; gap:16px; }
    .small { font-size:0.85rem; color: #666; }
    </style>
""", unsafe_allow_html=True)

# ---- STATIC DATA PATH ----
DATA_PATH = "Ubertrip/Uber-Jan-Feb-FOIL.csv"  # put your static CSV here

# ---- HELPERS ----
@st.cache_data
def load_and_prepare(path):
    df = pd.read_csv(path)

    # ---- Identify datetime column ----
    datetime_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            datetime_col = c
            break
    if datetime_col is None:
        raise ValueError("No datetime-like column found in CSV (expected something like 'date').")

    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col]).copy()
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # ---- Pick a count/target column ----
    count_col = None
    for c in df.columns:
        if c.lower() in ['base', 'id', 'tripid', 'trip_id', 'dispatching_base_num', 'trips', 'active_vehicles']:
            count_col = c
            break
    if count_col is None:
        # fallback to the first non-datetime column
        possible_cols = [c for c in df.columns if c != datetime_col]
        count_col = possible_cols[0] if possible_cols else datetime_col

    # ---- Build a daily or hourly time series ----
    df = df.set_index(datetime_col)
    freq = 'H' if df.index.inferred_freq != 'D' else 'D'

    # If the selected column is numeric, aggregate by sum
    if pd.api.types.is_numeric_dtype(df[count_col]):
        hourly = df[count_col].resample(freq).sum().rename('Count').to_frame()
    else:
        # Otherwise, just count entries per time period
        hourly = df[count_col].resample(freq).count().rename('Count').to_frame()

    hourly.index.name = 'Date'
    hourly = hourly.reset_index()
    return hourly

def create_lag_features(series, window_size):
    """
    series: 1d numpy array or pd.Series
    returns X (n_samples, window_size), y (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE - handle zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, np.nan, y_true))) * 100
    if np.isnan(mape):
        mape = float('nan')
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE_%': mape}

# ---- MAIN APP LAYOUT ----
st.title("🚕 Uber Trips Forecast — Clean UI")
st.markdown("A lightweight app to prepare hourly trip counts, train time-window models, and inspect performance. Put your static CSV in the same folder as `app.py` and run `streamlit run app.py`.")

# Left column: controls
left, right = st.columns((1, 2))
with left:
    st.header("Configuration")
    st.markdown("**Data**")
    st.write(f"`{DATA_PATH}` (static)")

    st.markdown("**Window & Split**")
    window_size = st.slider("Lag window (hours)", min_value=6, max_value=72, value=24, step=6,
                            help="Number of previous hours used as features to predict the next hour")
    split_mode = st.radio("Train/Test split style", options=["By % (chronological)", "By date cutoff (chronological)"], index=0)
    if split_mode.startswith("By %"):
        test_size_pct = st.slider("Test size (%)", 5, 50, 20, step=5)
    else:
        cutoff_date_str = st.text_input("Cutoff date (YYYY-MM-DD HH:MM) — test starts after this", value="2014-04-20 00:00")
    st.markdown("---")
    st.markdown("**Model**")
    available_models = ["Random Forest", "Gradient Boosting"]
    if XGBOOST_AVAILABLE:
        available_models.insert(0, "XGBoost")
    model_choice = st.selectbox("Select model", available_models)

    # Simple hyperparams (small)
    st.markdown("**Hyperparameters (basic)**")
    n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
    max_depth = st.slider("max_depth", 3, 30, 10, step=1)
    st.markdown("---")
    if st.button("Load data & preview"):
        st.session_state.load_data_preview = True

# Right column: data preview + EDA placeholders
with right:
    try:
        hourly = load_and_prepare(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. Place your CSV there and rerun.")
        st.stop()
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        st.stop()

    st.subheader("Hourly series preview")
    st.write(hourly.head(10))
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.lineplot(x='Date', y='Count', data=hourly, ax=ax)
    ax.set_title("Hourly Trip Counts (preview)")
    ax.set_xlabel("")
    st.pyplot(fig)

    # Summary stats
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Records (hours)", f"{hourly.shape[0]}")
    col_b.metric("Start", str(hourly['Date'].min()))
    col_c.metric("End", str(hourly['Date'].max()))

st.markdown("---")

# Build features
series = hourly['Count'].values
X_all, y_all = create_lag_features(series, window_size)
# Create a DataFrame for convenience (feature names lag_1 ... lag_n)
feature_names = [f"lag_{i}" for i in range(window_size, 0, -1)]
X_df = pd.DataFrame(X_all, columns=feature_names)
y_df = pd.Series(y_all, name='target')

# Provide option to inspect a sample
st.subheader("Prepared dataset for modeling")
st.write("Features (lag window) → predict next hour's count")
st.dataframe(pd.concat([X_df.head(6), y_df.head(6)], axis=1))

# Train/test split (chronological)
if split_mode.startswith("By %"):
    test_size = test_size_pct / 100.0
    split_index = int(len(X_df) * (1 - test_size))
    X_train = X_df.iloc[:split_index].values
    X_test = X_df.iloc[split_index:].values
    y_train = y_df.iloc[:split_index].values
    y_test = y_df.iloc[split_index:].values
    split_info = f"Chronological split: first {100-test_size_pct}% train, last {test_size_pct}% test"
else:
    try:
        cutoff_ts = pd.to_datetime(cutoff_date_str)
        # find index in the original hourly series where prediction time >= cutoff_ts
        # We used windows: the i-th sample predicts hour at index i+window_size
        dates_for_targets = hourly['Date'].iloc[window_size:].reset_index(drop=True)
        mask_train = dates_for_targets < cutoff_ts
        X_train = X_df.iloc[mask_train.values].values
        X_test = X_df.iloc[~mask_train.values].values
        y_train = y_df.iloc[mask_train.values].values
        y_test = y_df.iloc[~mask_train.values].values
        split_info = f"Chronological split by date cutoff: train before {cutoff_ts}, test from {cutoff_ts} onward"
    except Exception as e:
        st.error(f"Invalid cutoff date: {e}")
        st.stop()

st.info(split_info + f" — Train samples: {len(y_train)}, Test samples: {len(y_test)}")

# TRAINING section
st.subheader("Train model")
train_col, perf_col = st.columns([1, 2])

with train_col:
    if st.button("Train model now"):
        # Train
        with st.spinner("Training model... this may take a little while depending on model/hyperparams"):
            t0 = time.time()
            if model_choice == "XGBoost":
                if not XGBOOST_AVAILABLE:
                    st.error("XGBoost is not available in the environment. Install `xgboost` in requirements and rerun.")
                else:
                    model = xgb.XGBRegressor(objective='reg:squarederror',
                                             n_estimators=n_estimators, max_depth=max_depth,
                                             random_state=42, verbosity=0)
                    model.fit(X_train, y_train)
            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
            else:  # Gradient Boosting (scikit-learn)
                model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
            t1 = time.time()
            # predict
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train) if len(X_train) > 0 else np.array([])
            # metrics
            metrics_test = compute_metrics(y_test, y_pred_test)
            metrics_train = compute_metrics(y_train, y_pred_train) if len(X_train) > 0 else {}
            # store in session state for later performance page
            st.session_state['latest_model'] = model
            st.session_state['metrics_test'] = metrics_test
            st.session_state['metrics_train'] = metrics_train
            st.session_state['y_test'] = y_test
            st.session_state['y_pred_test'] = y_pred_test
            st.session_state['y_train'] = y_train
            st.session_state['y_pred_train'] = y_pred_train
            st.session_state['feature_names'] = feature_names
            st.success(f"Training finished in {t1-t0:.1f}s — model: {model_choice}")

with perf_col:
    st.subheader("Recent training summary")
    if 'metrics_test' in st.session_state:
        mt = st.session_state['metrics_test']
        ml = st.session_state['metrics_train']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test MAE", f"{mt['MAE']:.2f}")
        col2.metric("Test RMSE", f"{mt['RMSE']:.2f}")
        col3.metric("Test R²", f"{mt['R2']:.3f}")
        col4.metric("Test MAPE", f"{mt['MAPE_%']:.2f} %")
        st.write("Tip: If MAPE shows NaN, it means there are zeros in the true counts — interpret accordingly.")
    else:
        st.info("Train a model to see performance metrics here.")

st.markdown("---")

# PERFORMANCE PAGE (visuals)
st.subheader("Performance & Visuals")
if 'latest_model' not in st.session_state:
    st.info("Train a model above to see plots and feature importance.")
else:
    model = st.session_state['latest_model']
    y_test = st.session_state['y_test']
    y_pred_test = st.session_state['y_pred_test']
    y_train = st.session_state.get('y_train', np.array([]))
    y_pred_train = st.session_state.get('y_pred_train', np.array([]))

    # 1) Actual vs Predicted plot (test)
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    # plot a window of test series to keep it readable: last 200 points or all if smaller
    nplot = min(len(y_test), 800)
    ax1.plot(y_test[-nplot:], label='Actual (test)', lw=1)
    ax1.plot(y_pred_test[-nplot:], label='Predicted (test)', lw=1)
    ax1.set_title("Actual vs Predicted on Test Set (last {} points)".format(nplot))
    ax1.legend()
    st.pyplot(fig1)

    # 2) Scatter actual vs predicted
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(y_test, y_pred_test, alpha=0.3, s=8)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Actual vs Predicted Scatter (test)")
    st.pyplot(fig2)

    # 3) Residuals histogram
    residuals = y_test - y_pred_test
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    sns.histplot(residuals, bins=40, kde=True, ax=ax3)
    ax3.set_title("Residuals Distribution (test)")
    st.pyplot(fig3)

    # 4) Feature importances (if available)
    st.subheader("Feature importances")
    importance_fig = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fn = st.session_state['feature_names']
            imp_df = pd.Series(importances, index=fn).sort_values(ascending=False).head(20)
            fig4, ax4 = plt.subplots(figsize=(8, min(6, 0.25*len(imp_df))))
            sns.barplot(x=imp_df.values, y=imp_df.index, ax=ax4)
            ax4.set_title("Top feature importances")
            st.pyplot(fig4)
        else:
            st.info("Model does not expose `feature_importances_` (e.g., XGBoost wrapper might require `.get_booster()` calls).")
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

    # 5) Show numeric metrics table
    st.subheader("Metrics (train vs test)")
    mtrain = st.session_state.get('metrics_train', {})
    mtest = st.session_state.get('metrics_test', {})
    metrics_df = pd.DataFrame({
        'metric': ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE_%'],
        'train': [mtrain.get('MAE', np.nan), mtrain.get('MSE', np.nan), mtrain.get('RMSE', np.nan), mtrain.get('R2', np.nan), mtrain.get('MAPE_%', np.nan)],
        'test': [mtest.get('MAE', np.nan), mtest.get('MSE', np.nan), mtest.get('RMSE', np.nan), mtest.get('R2', np.nan), mtest.get('MAPE_%', np.nan)]
    })
    st.table(metrics_df.set_index('metric').style.format("{:.3f}"))

st.markdown("---")
st.caption("Tip: This simple window-based ML approach is a good baseline. For production forecasting consider time-series specific models (ARIMA, Prophet) or sequence models (LSTM) and careful cross-validation like TimeSeriesSplit.")



