import requests
from datetime import datetime, timedelta
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Mumbai PM2.5 Forecast",
    page_icon="🌫️",
    layout="wide"
)


@st.cache_data(ttl=900)
def fetch_recent_pm25():
    api_key = st.secrets.get("AQI_API_KEY", "")

    if not api_key:
        return None, "AQI_API_KEY is missing from Streamlit Secrets."

    url = "https://airquality.xkdr.org/v1/measurements"
    headers = {"Authorization": f"Bearer {api_key}"}

    params = {
        "city": "Mumbai",
        "parameter": "PM2.5",
        "agg": "hourly",
        "format": "csv"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()

        from io import StringIO
        live_df = pd.read_csv(StringIO(response.text))

        if live_df.empty:
            return None, "The API returned no recent PM2.5 readings."

        live_df["period_start"] = pd.to_datetime(
            live_df["period_start"], errors="coerce"
        )
        live_df["mean"] = pd.to_numeric(
            live_df["mean"], errors="coerce"
        )
        live_df = live_df.dropna(
            subset=["period_start", "mean"]
        ).sort_values("period_start")

        if live_df.empty:
            return None, "No valid readings were returned."

        return live_df, None

    except Exception as exc:
        return None, f"Could not retrieve recent readings: {exc}"

st.title("Mumbai PM2.5 Forecasting Dashboard")
st.caption("Hourly air pollution analysis and machine learning estimates")

# Load project files
@st.cache_data
def load_data():
    data = pd.read_csv(
        "mumbai_pm25_hourly.csv",
        parse_dates=["period_start"]
    )
    stations = pd.read_csv("stations.csv")

    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data = data.dropna(
        subset=["station_id", "period_start", "mean"]
    )
    data = data[data["mean"] >= 0]
    data = data.sort_values(["station_id", "period_start"])
    data = data.drop_duplicates(
        ["station_id", "period_start"],
        keep="last"
    )
    return data, stations


@st.cache_resource
def load_model():
    with open("model_config.json", "r") as f:
        config = json.load(f)
    model = joblib.load("pm25_model.joblib")
    return model, config


try:
    data, stations = load_data()
    model, config = load_model()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

station_names = (
    stations.drop_duplicates("station_id")
    .set_index("station_id")["station_name"]
    .to_dict()
)

station_ids = sorted(data["station_id"].unique())

selected_station = st.sidebar.selectbox(
    "Choose monitoring station",
    station_ids,
    format_func=lambda x: f"{x} — {station_names.get(x, x)}"
)

station_data = (
    data[data["station_id"] == selected_station]
    .sort_values("period_start")
    .copy()
)

if station_data.empty:
    st.warning("No data available for this station.")
    st.stop()

latest = station_data.iloc[-1]

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Latest observed PM2.5", f"{latest['mean']:.1f} µg/m³")
c2.metric("Station average", f"{station_data['mean'].mean():.1f} µg/m³")
c3.metric("Total observations", f"{len(station_data):,}")

st.info(
    f"Latest observation: {latest['period_start']}. "
    "The dataset ends in September 2025; this is not a live air-quality feed."
)

# Historical chart

# Historical chart
st.subheader("Historical PM2.5 readings")

min_date = station_data["period_start"].min().date()
max_date = station_data["period_start"].max().date()

default_start = max(
    min_date,
    max_date - pd.Timedelta(days=30)
)

date_range = st.sidebar.date_input(
    "Choose chart date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.warning("Please select both a start and an end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

chart_df = station_data[
    (station_data["period_start"].dt.date >= start_date)
    & (station_data["period_start"].dt.date <= end_date)
].copy()

if chart_df.empty:
    st.warning("No observations are available for this date range.")
else:
    # Keep missing hours as gaps rather than connecting across them.
    chart_df = chart_df.set_index("period_start").sort_index()
    hourly_values = chart_df["mean"].resample("h").mean()

    st.caption(
        f"{chart_df['mean'].count():,} observations in the selected period. "
        "Blank gaps indicate hours without recorded readings."
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        hourly_values.index,
        hourly_values.values,
        marker=".",
        markersize=4,
        linewidth=1
    )
    ax.set_title("Hourly PM2.5 concentration")
    ax.set_xlabel("Date and time")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

# Daily average trend
st.subheader("Daily average PM2.5")

daily_df = (
    chart_df["mean"]
    .resample("D")
    .mean()
    .dropna()
)

if daily_df.empty:
    st.info("There are no daily averages for this date range.")
else:
    st.line_chart(
        daily_df,
        x_label="Date",
        y_label="Average PM2.5 (µg/m³)"
    )


# One-hour forecast
st.subheader("Next-hour PM2.5 estimate")

station_lookup = station_data.set_index("period_start")["mean"]
last_time = station_data["period_start"].max()
next_time = last_time + pd.Timedelta(hours=1)

def get_exact_lag(hours):
    target_time = next_time - pd.Timedelta(hours=hours)
    return station_lookup.get(target_time, np.nan)

lags = {
    1: get_exact_lag(1),
    2: get_exact_lag(2),
    3: get_exact_lag(3),
    24: get_exact_lag(24)
}

station_to_code = {
    str(k): int(v)
    for k, v in config["station_to_code"].items()
}

station_code = station_to_code.get(str(selected_station))

if station_code is None:
    st.warning("This station is not present in the saved model configuration.")
elif pd.isna(lags[1]):
    st.warning(
        "No observation exists exactly one hour before the forecast time. "
        "A forecast cannot be produced reliably."
    )
else:
    features = pd.DataFrame([{
        "station_code": station_code,
        "lag_1": lags[1],
        "lag_2": lags[2],
        "lag_3": lags[3],
        "lag_24": lags[24],
        "hour": next_time.hour,
        "day_of_week": next_time.dayofweek,
        "month": next_time.month
    }], columns=config["feature_columns"])

    prediction = max(0, float(model.predict(features)[0]))

    st.metric(
        f"Estimated PM2.5 at {next_time}",
        f"{prediction:.1f} µg/m³"
    )
    st.caption(
        "Historical model estimate for the hour after the latest observation, "
        "not a forecast for the current date."
    )

# Historical test evaluation
st.subheader("Model performance")

cutoff = pd.Timestamp(config["training_cutoff"])

test_data = data[data["period_start"] >= cutoff].copy()
groups = {
    sid: group.set_index("period_start")["mean"]
    for sid, group in test_data.groupby("station_id")
}

rows = []

for row in test_data.itertuples(index=False):
    timestamp = row.period_start
    series = groups[row.station_id]

    lag_values = {
        h: series.get(timestamp - pd.Timedelta(hours=h), np.nan)
        for h in [1, 2, 3, 24]
    }

    if pd.isna(lag_values[1]):
        continue

    code = station_to_code.get(str(row.station_id))
    if code is None:
        continue

    rows.append({
        "actual": row.mean,
        "station_code": code,
        "lag_1": lag_values[1],
        "lag_2": lag_values[2],
        "lag_3": lag_values[3],
        "lag_24": lag_values[24],
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "month": timestamp.month
    })

if rows:
    evaluation = pd.DataFrame(rows)
    X_eval = evaluation[config["feature_columns"]]
    actual = evaluation["actual"].to_numpy()
    predicted = model.predict(X_eval)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f} µg/m³")
    c2.metric("RMSE", f"{rmse:.2f} µg/m³")
    c3.metric("R²", f"{r2:.3f}")

    baseline_mae = mean_absolute_error(
        actual, evaluation["lag_1"].to_numpy()
    )
    st.write(f"Previous-hour baseline MAE: {baseline_mae:.2f} µg/m³")
else:
    st.warning("Not enough consecutive test observations to calculate metrics.")

# Station details
st.subheader("Monitoring station details")
details = stations[
    stations["station_id"].astype(str) == str(selected_station)
]
st.dataframe(details, use_container_width=True)

st.caption(
    "Source: CPCB observations via the India Air Quality Database. "
    "Predictions are estimates and are not official CPCB forecasts."
)
