import logging
import os

logging.getLogger("pystan").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("statsmodels").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
import statsmodels.api as sm
import warnings
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from shiny import *
from shiny.express import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

# ── Load & split data ─────────────────────────────────────────────────────────
df = pd.read_csv("bills.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.rename(columns={"Date": "ds", "Amount": "y"})

dfpse     = df[df["Type"] == "PSE"].set_index("ds").asfreq("ME")
dfwater   = df[df["Type"] == "water"].set_index("ds").asfreq("2ME")
dfgarbage = df[df["Type"] == "garbage"].set_index("ds").asfreq("QE")

# Billing frequency labels and month offsets
FREQ_MAP = {"PSE": "ME", "water": "2ME", "garbage": "QE"}
PERIOD_MONTHS = {"PSE": 1, "water": 2, "garbage": 3}

# ── SARIMAX helpers ────────────────────────────────────────────────────────────
def fit_sarimax(series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Fit SARIMAX and return the fitted result."""
    series = series["y"].dropna()
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

def forecast_sarimax(fitted, series, n_periods, freq):
    """Return historical fit + future forecast as a single DataFrame."""
    hist = fitted.fittedvalues
    pred = fitted.get_forecast(steps=n_periods)
    future_mean = pred.predicted_mean
    ci = pred.conf_int()

    # Build a combined index
    last_date = series.index[-1]
    future_idx = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[1:]
    future_mean.index = future_idx
    ci.index = future_idx

    return hist, future_mean, ci, series["y"].dropna()

def plot_forecast(hist_actual, hist_fitted, future_mean, ci, title, ax):
    """Plot historical actuals, fitted line, future forecast with CI band."""
    ax.plot(hist_actual.index, hist_actual.values, color="#333333", linewidth=1.5, label="Actual")
    ax.plot(hist_fitted.index, hist_fitted.values, color="#1f77b4", linewidth=1.2, linestyle="--", label="Fitted", alpha=0.8)
    ax.plot(future_mean.index, future_mean.values, color="#EF9F27", linewidth=2, label="Forecast")
    ax.fill_between(
        ci.index,
        ci.iloc[:, 0],
        ci.iloc[:, 1],
        color="#EF9F27",
        alpha=0.2,
        label="95% CI",
    )
    ax.axvline(x=hist_actual.index[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount ($)")
    ax.legend(fontsize=8)

# ── Pre-fit seasonal orders (tuned for each utility) ─────────────────────────
# PSE: monthly billing, strong yearly seasonality
ORDERS = {
    "PSE":     ((1,1,1), (1,1,1,12)),
    "water":   ((1,1,1), (1,1,1,6)),   # bi-monthly, 6-period yearly cycle
    "garbage": ((1,1,1), (1,1,1,4)),   # quarterly, 4-period yearly cycle
}

# ── UI ─────────────────────────────────────────────────────────────────────────
ui.page_opts(title="Utility Bill Forecasting and Anomaly Detection", fillable=True)

with ui.sidebar():
    ui.input_date("date_start", "Start Date:", value="2019-01-19")
    ui.input_date("date_end",   "End Date:",   value=str(pd.Timestamp.now().date()))

with ui.navset_pill(id="tab"):

    # ── Tab 1: Time Series Analysis ───────────────────────────────────────────
    with ui.nav_panel("Time Series Analysis"):

        with ui.card(full_screen=True):
            ui.card_header("Historical Seasonality Trends of Utility Bill Amounts")
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_select("Type", "Select a Utility Category:",
                                    ["PSE", "garbage", "water"], selected="PSE")
                    ui.input_select("seasonality", "Choose seasonality:",
                                    ["Year", "Month"], selected="Month")

                @render.plot()
                def seasonality_plot():
                    sel   = input.Type()
                    seas  = input.seasonality()
                    start = pd.to_datetime(input.date_start())
                    end   = pd.to_datetime(input.date_end())

                    raw = df[df["Type"] == sel]
                    raw = raw[(raw["ds"] >= start) & (raw["ds"] <= end)]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    if seas == "Year":
                        agg = raw.groupby(raw["ds"].dt.year)["y"].mean()
                        sns.barplot(x=agg.index, y=agg.values, ax=ax)
                        ax.set_xlabel("Year")
                    else:
                        agg = raw.groupby(raw["ds"].dt.month)["y"].mean()
                        sns.barplot(x=agg.index, y=agg.values, ax=ax)
                        ax.set_xlabel("Month")
                    ax.set_ylabel("Average Bill Amount ($)")
                    return fig

        with ui.card(full_screen=True):
            ui.card_header(
                "Utility Bill Forecast — PSE: monthly · Water: bi-monthly · Garbage: quarterly"
            )
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_select("utility_type", "Select a Utility Category:",
                                    ["PSE", "garbage", "water"], selected="PSE")
                    ui.input_select("timeframe", "Future forecast billing periods:",
                                    ["1 Period", "3 Periods", "6 Periods",
                                     "9 Periods", "12 Periods"], selected="1 Period")

                @reactive.calc
                def filtered_series():
                    start = pd.to_datetime(input.date_start())
                    end   = pd.to_datetime(input.date_end())
                    ut    = input.utility_type()
                    freq  = FREQ_MAP[ut]
                    raw   = df[df["Type"] == ut].copy()
                    raw   = raw[(raw["ds"] >= start) & (raw["ds"] <= end)]
                    raw   = raw.set_index("ds").asfreq(freq)
                    return raw, freq, ut

                @reactive.calc
                def build_model():
                    series, freq, ut = filtered_series()
                    order, seasonal_order = ORDERS[ut]
                    return fit_sarimax(series, order=order, seasonal_order=seasonal_order), series, freq, ut

                @render.plot()
                def forecast_plot():
                    fitted, series, freq, ut = build_model()
                    n_periods = {"1 Period": 1, "3 Periods": 3, "6 Periods": 6,
                                 "9 Periods": 9, "12 Periods": 12}[input.timeframe()]

                    hist_fitted, future_mean, ci, hist_actual = forecast_sarimax(
                        fitted, series, n_periods, freq
                    )

                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_forecast(hist_actual, hist_fitted, future_mean, ci,
                                  f"{ut} Bill Forecast ({n_periods} period{'s' if n_periods > 1 else ''} ahead)",
                                  ax)
                    fig.tight_layout()
                    return fig

        with ui.card(full_screen=True):
            ui.card_header("Forecast Error Metrics")
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_checkbox("show_metrics", "Show error metrics", False)

                @render.data_frame
                def error_table():
                    if input.show_metrics():
                        fitted, series, freq, ut = build_model()
                        actual  = series["y"].dropna()
                        fitted_vals = fitted.fittedvalues.reindex(actual.index).dropna()
                        actual_aligned = actual.reindex(fitted_vals.index)
                        resid   = actual_aligned - fitted_vals
                        mae     = np.mean(np.abs(resid))
                        rmse    = np.sqrt(np.mean(resid**2))
                        mape    = np.mean(np.abs(resid / actual_aligned.replace(0, np.nan))) * 100
                        metrics = pd.DataFrame({
                            "Metric": ["MAE", "RMSE", "MAPE (%)"],
                            "Value":  [round(mae, 2), round(rmse, 2), round(mape, 2)],
                        })
                        return render.DataTable(metrics)

    # ── Tab 2: Anomaly Detection ──────────────────────────────────────────────
    with ui.nav_panel("Anomaly Detection"):
        with ui.card():
            ui.card_header("Detect Anomalies on entire Dataset")
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_radio_buttons(
                        "only_anomalies", "Anomaly View",
                        {"a": "All Data with Anomalies", "b": "Only Anomalies"}
                    )

                @reactive.calc
                def data():
                    dfc = pd.read_csv("bills.csv")
                    dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce")
                    dfc.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
                    dfc["Date"] = pd.to_datetime(dfc["Date"]).astype("datetime64[ns]")
                    dfc["Count"] = 1
                    dfc["date_year"]      = dfc["Date"].dt.year
                    dfc["date_month_num"] = dfc["Date"].dt.month

                    nonnumeric_columns = dfc.select_dtypes("object").copy()
                    nonnumeric_columns["Type"] = dfc["Type"]
                    for col in nonnumeric_columns:
                        dfc[col].fillna(0, inplace=True)
                    dfc["Amount"].fillna(0, inplace=True)
                    dfc = dfc.drop_duplicates()
                    df1 = dfc.copy()

                    df_dummies = pd.get_dummies(
                        pd.DataFrame(nonnumeric_columns), drop_first=True
                    ).astype(int)
                    dfc = dfc.drop(nonnumeric_columns.columns, axis=1)

                    df_numerics = dfc[["Amount", "People", "perPerson", "Count"]]
                    scaler      = StandardScaler()
                    scaled_data = scaler.fit_transform(df_numerics)
                    scaled_df   = pd.DataFrame(scaled_data, columns=df_numerics.columns)
                    scaled_df   = scaled_df.rename(columns={
                        "Amount":    "Amount_scaled",
                        "perPerson": "perPerson_scaled",
                        "People":    "People_scaled",
                    })

                    dff = pd.concat([scaled_df, df_dummies], axis=1)
                    dfs = pd.concat([dfc, scaled_df, df_dummies], axis=1)

                    # 1. Z-Score
                    mean   = statistics.mean(dfs.Amount_scaled)
                    stdev  = statistics.stdev(dfs.Amount_scaled)
                    dfs["Z_score"]        = [(x - mean) / stdev for x in dfs.Amount_scaled]
                    dfs["ZScore_anomaly"] = (np.abs(dfs["Amount_scaled"]) > 3).astype(int)

                    # 2. Z-Score on Max
                    df_def = df_numerics[["Amount", "perPerson"]].copy()
                    z_max  = np.abs((df_def - df_def.mean()) / df_def.std())
                    dfs["zscore_maxAmount"]         = z_max.max(axis=1)
                    dfs["zscore_maxAmount_anomaly"] = (np.abs(dfs["zscore_maxAmount"]) > 3).astype(int)

                    # 3. Local Outlier Factor
                    lof          = LocalOutlierFactor(n_neighbors=105, contamination=0.005)
                    dfs["lof"]   = lof.fit_predict(dff)
                    dfs["lof_anomaly"] = (dfs["lof"] == -1).astype(int)

                    # 4. Isolation Forest
                    iso_forest = IsolationForest(n_estimators=105, contamination=0.005)
                    iso_forest.fit(dff)
                    dfs["isolationforest"]         = iso_forest.predict(dff)
                    dfs["isolationforest_anomaly"] = (dfs["isolationforest"] == -1).astype(int)

                    # 5. MLP Autoencoder
                    dff_values  = dff.values.astype(np.float32)
                    autoencoder = MLPRegressor(
                        hidden_layer_sizes=(20, 10, 10, 20),
                        activation="tanh", solver="adam",
                        max_iter=200, random_state=42, verbose=False,
                    )
                    autoencoder.fit(dff_values, dff_values)
                    reconstructions       = autoencoder.predict(dff_values)
                    mse                   = np.mean(np.power(dff_values - reconstructions, 2), axis=1)
                    dfs["autoencoders"]   = mse
                    threshold             = np.percentile(mse, 99)
                    dfs["autoencoder_anomaly"] = (mse > threshold).astype(int)

                    # 6. Combined
                    dfs["combined_anomalies"] = (
                        (dfs["Z_score"] + dfs["lof"] + dfs["isolationforest"] + dfs["autoencoders"]) / 4
                    )
                    dfs["combined_anomalies"] = (np.abs(dfs["combined_anomalies"]) > 3).astype(int)

                    # Annotate df1
                    df1["ZScore_anomaly"]           = dfs["ZScore_anomaly"].values
                    df1["zscore_maxDefRej_anomaly"] = dfs["zscore_maxAmount_anomaly"].values
                    df1["lof_anomaly"]              = dfs["lof_anomaly"].values
                    df1["isolationforest_anomaly"]  = dfs["isolationforest_anomaly"].values
                    df1["autoencoder_anomaly"]      = dfs["autoencoder_anomaly"].values
                    df1["combined_anomalies"]       = dfs["combined_anomalies"].values

                    df1["anomaly_indicator"] = df1[[
                        "ZScore_anomaly", "zscore_maxDefRej_anomaly", "lof_anomaly",
                        "isolationforest_anomaly", "autoencoder_anomaly", "combined_anomalies"
                    ]].sum(axis=1) > 0

                    df1["a_type"] = 0
                    df1.loc[df1.ZScore_anomaly == 1,          "a_type"] = 1
                    df1.loc[df1.zscore_maxDefRej_anomaly == 1, "a_type"] = 2
                    df1.loc[df1.lof_anomaly == 1,             "a_type"] = 3
                    df1.loc[df1.isolationforest_anomaly == 1, "a_type"] = 4
                    df1.loc[df1.autoencoder_anomaly == 1,     "a_type"] = 5
                    df1.loc[df1.combined_anomalies == 1,      "a_type"] = 7

                    df1["a_name"] = "None"
                    df1.loc[df1.ZScore_anomaly == 1,          "a_name"] = "ZScore"
                    df1.loc[df1.zscore_maxDefRej_anomaly == 1, "a_name"] = "ZScore Max"
                    df1.loc[df1.lof_anomaly == 1,             "a_name"] = "Local Outlier Factor"
                    df1.loc[df1.isolationforest_anomaly == 1, "a_name"] = "Isolation Forest"
                    df1.loc[df1.autoencoder_anomaly == 1,     "a_name"] = "AutoEncoder"
                    df1.loc[df1.combined_anomalies == 1,      "a_name"] = "Multiple"

                    return df1

                @render.plot()
                def scatter_plot():
                    df1 = data()
                    if df1.empty:
                        return None
                    fig, ax = plt.subplots(figsize=(12, 5))
                    if input.only_anomalies() == "a":
                        ax.scatter(df1["Date"], df1["Amount"], c="black", s=12, label="Normal")
                        anomalies = df1[df1["anomaly_indicator"]]
                        sc = ax.scatter(anomalies["Date"], anomalies["Amount"],
                                        c=anomalies["a_type"], s=anomalies["a_type"] * 20,
                                        cmap="tab10", zorder=5, label="Anomaly")
                        plt.colorbar(sc, ax=ax, label="Anomaly Type")
                        ax.set_title("Dataset with Anomalies")
                    else:
                        anomalies = df1[df1["anomaly_indicator"]]
                        sc = ax.scatter(anomalies["Date"], anomalies["Amount"],
                                        c=anomalies["a_type"], s=anomalies["a_type"] * 20,
                                        cmap="tab10", zorder=5)
                        plt.colorbar(sc, ax=ax, label="Anomaly Type")
                        ax.set_title("Only Anomalies")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Amount ($)")
                    fig.tight_layout()
                    return fig

                @render.text
                def detail_data():
                    return "Detail Data for Anomalies"

                @render.data_frame
                def detail_data_frame():
                    df1 = data()
                    cols = ["ZScore_anomaly", "zscore_maxDefRej_anomaly", "lof_anomaly",
                            "isolationforest_anomaly", "autoencoder_anomaly", "combined_anomalies"]
                    out = df1[df1[cols].sum(axis=1) > 0]
                    if out.empty:
                        return None
                    return render.DataGrid(pd.DataFrame(out))