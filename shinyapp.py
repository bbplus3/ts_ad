import logging
import os

logging.getLogger("pystan").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statistics
import statsmodels.api as sm
from datetime import datetime as dt, timedelta, date
from dateutil import relativedelta
from matplotlib.backends.backend_agg import RendererAgg
from pathlib import Path
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from shiny import *
from shiny.express import *
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tqdm
import itertools
plt.style.use('fivethirtyeight')

# python -m shiny run shinyapp.py --port 8080

# Load Data
df = pd.read_csv("bills.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.rename(columns={"Date": "ds", "Amount": "y"})

# Subset Data
dfpse = df[df["Type"] == "PSE"]
dfwater = df[df["Type"] == "water"].copy()
dfwater['cap'] = dfwater['y'].max() * 1.1  # Adding capacity for logistic growth
dfgarbage = df[df["Type"] == "garbage"]

# Define UI
ui.page_opts(title="Utility Bill Forecasting and Anomaly Detection", fillable=True)

with ui.sidebar():
    ui.input_date("date_start", "Start Date:", value='2019-01-19')
    ui.input_date("date_end", "End Date:", value=pd.Timestamp.now())

with ui.navset_pill(id="tab"):
    with ui.nav_panel("Time Series Analysis"):
        with ui.card(full_screen=True):
            ui.card_header("Historical Seasonality Trends of Utility Bill Amounts")

            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_select("Type", "Select a Utility Category:", ["PSE", "garbage", "water"], selected="PSE")
                    ui.input_select("seasonality", "Choose seasonality:", ["Year", "Month"], selected="Month")

                @render.plot()
                def seasonality_plot():
                    selected_type = input.Type()
                    selected_seasonality = input.seasonality()
                    start_date = pd.to_datetime(input.date_start())
                    end_date = pd.to_datetime(input.date_end())

                    df_selected = {"PSE": dfpse, "water": dfwater, "garbage": dfgarbage}.get(selected_type, dfpse)
                    df_selected = df_selected[(df_selected['ds'] >= start_date) & (df_selected['ds'] <= end_date)]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    if selected_seasonality == "Year":
                        yearAggregated = df_selected.groupby(df_selected['ds'].dt.year)["y"].mean()
                        sns.barplot(x=yearAggregated.index, y=yearAggregated.values, ax=ax)
                    else:
                        monthAggregated = df_selected.groupby(df_selected['ds'].dt.month)["y"].mean()
                        sns.barplot(x=monthAggregated.index, y=monthAggregated.values, ax=ax)

                    ax.set_xlabel("Time Period")
                    ax.set_ylabel("Average Bill Amount")
                    return fig

        with ui.card(full_screen=True):
            with ui.card(full_screen=True):
                ui.card_header("Utility Bill Forecast: billing period frequency is 1 month for PSE, 2 months for water, 3 months for garbage")

                with ui.layout_sidebar():
                    with ui.sidebar(bg="#f8f8f8"):
                        ui.input_select("utility_type", "Select a Utility Category:", ["PSE", "garbage", "water"], selected="PSE")
                        ui.input_select("timeframe", "Choose future forecast billing periods:", ["1 Period", "3 Periods", "6 Periods", "9 Periods", "12 Periods"], selected="1 Period")

                @reactive.calc
                def filtered_data():
                    start_date = pd.to_datetime(input.date_start())
                    end_date = pd.to_datetime(input.date_end())
                    df_selected = {"PSE": dfpse, "water": dfwater, "garbage": dfgarbage}.get(input.utility_type(), dfpse)
                    return df_selected[(df_selected['ds'] >= start_date) & (df_selected['ds'] <= end_date)]

                @reactive.calc
                def build_models():
                    df_pse = filtered_data() if input.utility_type() == "PSE" else dfpse
                    df_water = filtered_data() if input.utility_type() == "water" else dfwater
                    df_garbage = filtered_data() if input.utility_type() == "garbage" else dfgarbage

                    model_pse = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.005, seasonality_prior_scale=0.01, seasonality_mode='additive', growth='linear')
                    model_pse.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    model_pse.add_seasonality(name='yearly', period=365.25, fourier_order=10)

                    model_water = Prophet(yearly_seasonality=False, changepoint_prior_scale=4, seasonality_prior_scale=0.1, seasonality_mode='additive', growth='logistic')
                    model_water.add_seasonality(name='bi_monthly', period=2*30.5, fourier_order=5)
                    model_water.add_seasonality(name='yearly', period=365.25, fourier_order=10)
                    for col in dfwater.columns:
                        if col.startswith("month_"):
                            model_water.add_regressor(col)

                    model_garbage = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05, seasonality_prior_scale=0.07, seasonality_mode='multiplicative', growth='linear')
                    model_garbage.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    model_garbage.add_seasonality(name='yearly', period=365.25, fourier_order=10)

                    model_pse.fit(df_pse)
                    model_water.fit(df_water)
                    model_garbage.fit(df_garbage)

                    return model_pse, model_water, model_garbage

                @render.plot()
                def forecast_plot():
                    model_pse, model_water, model_garbage = build_models()
                    forecast_periods = {"1 Period": 1, "3 Periods": 3, "6 Periods": 6, "9 Periods": 9, "12 Periods": 12}
                    future_periods = forecast_periods[input.timeframe()]

                    if input.utility_type() == "PSE":
                        model = model_pse
                        future = model.make_future_dataframe(periods=future_periods, freq='ME')
                        forecast = model.predict(future)
                    elif input.utility_type() == "water":
                        model = model_water
                        future = model.make_future_dataframe(periods=future_periods, freq='2M')
                        future['cap'] = dfwater['cap'].max()
                        forecast = model.predict(future)
                    else:
                        model = model_garbage
                        future = model.make_future_dataframe(periods=future_periods, freq='3M')
                        forecast = model.predict(future)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    model.plot(forecast, ax=ax)
                    return fig

        with ui.card(full_screen=True):
            ui.card_header("Error Metrics for Forecast Model")
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_checkbox("show_plot", "Show me the metrics", False)

            @render.plot()
            def model_evaluation():
                if input.show_plot():
                    model_pse, model_water, model_garbage = build_models()
                    if input.utility_type() == "PSE":
                        model = model_pse
                    elif input.utility_type() == "water":
                        model = model_water
                    else:
                        model = model_garbage
                    df_cv = cross_validation(model, initial="720 days", period="90 days", horizon="90 days")
                    fig = plot_cross_validation_metric(df_cv, metric="mape")
                    return fig

            @render.data_frame
            def error_table():
                if input.show_plot():
                    model_pse, model_water, model_garbage = build_models()
                    if input.utility_type() == "PSE":
                        model = model_pse
                    elif input.utility_type() == "water":
                        model = model_water
                    else:
                        model = model_garbage
                    df_cv = cross_validation(model, initial="720 days", period="90 days", horizon="90 days")
                    df_perf = performance_metrics(df_cv)
                    return render.DataTable(df_perf)

    with ui.nav_panel("Anomaly Detection"):

        with ui.card():
            ui.card_header("Detect Anomalies on entire Dataset (not editable with date input on sidebar)")
            with ui.layout_sidebar():
                with ui.sidebar(bg="#f8f8f8"):
                    ui.input_radio_buttons("only_anomalies", "Anomaly View", {"a": "All Data with Anomalies", "b": "Only Anomalies"})

            @reactive.calc
            def data():
                df = pd.read_csv("bills.csv")
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
                df.replace('<NA>', pd.NA, inplace=True)

                df['Date'] = pd.to_datetime(df['Date']).astype('datetime64[ns]')
                df['Count'] = 1
                df['date_year'] = df['Date'].dt.year
                df['date_month_num'] = df['Date'].dt.month

                nonnumeric_columns = df.select_dtypes('object')
                nonnumeric_columns['Type'] = df['Type']

                for col in nonnumeric_columns:
                    df[col].fillna(0, inplace=True)

                df['Amount'].fillna(0, inplace=True)
                df = df.drop_duplicates()
                df1 = df.copy()

                nonnumeric_columns = pd.DataFrame(nonnumeric_columns)
                df_dummies = pd.get_dummies(nonnumeric_columns, drop_first=True)
                df_dummies = df_dummies.astype(int)
                df = df.drop(nonnumeric_columns.columns, axis=1)

                df_numerics = df[['Amount', 'People', 'perPerson', 'Count']]
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df_numerics)
                scaled_df = pd.DataFrame(scaled_data, columns=df_numerics.columns)
                scaled_df = scaled_df.rename(columns={'Amount': 'Amount_scaled', 'perPerson': 'perPerson_scaled', 'People': 'People_scaled'})

                dff = pd.concat([scaled_df, df_dummies], axis=1)
                dfs = pd.concat([df, scaled_df, df_dummies], axis=1)

                ########################### 1. ZScore ###########################
                mean = statistics.mean(dfs.Amount_scaled)
                stdev = statistics.stdev(dfs.Amount_scaled)
                values = np.array(dfs.Amount_scaled)
                zscore = [(x - mean) / stdev for x in values]
                dfs['Z_score'] = zscore
                dfs['ZScore_anomaly'] = (np.abs(dfs['Amount_scaled']) > 3).astype(int)

                ########################### 2. ZScore on Max ###########################
                df_def = df_numerics[['Amount', 'perPerson']].copy()
                z_scores = np.abs((df_def - df_def.mean()) / df_def.std())
                dfs['zscore_maxAmount'] = z_scores.max(axis=1)
                dfs['zscore_maxAmount_anomaly'] = (np.abs(dfs['zscore_maxAmount']) > 3).astype(int)

                ########################### 3. Local Outlier Factor ###########################
                lof = LocalOutlierFactor(n_neighbors=105, contamination=0.005)
                dfs['lof'] = lof.fit_predict(dff)
                dfs['lof_anomaly'] = dfs['lof'].apply(lambda x: 1 if x == -1 else 0)

                ########################### 4. Isolation Forest ###########################
                iso_forest = IsolationForest(n_estimators=105, contamination=0.005)
                iso_forest.fit(dff)
                dfs['isolationforest'] = iso_forest.predict(dff)
                dfs['isolationforest_anomaly'] = dfs['isolationforest'].apply(lambda x: 1 if x == -1 else 0)

                ########################### 5. Autoencoder (sklearn MLP) ###########################
                # MLPRegressor as a lightweight autoencoder replacement —
                # trained to reconstruct its own input; high reconstruction
                # error indicates anomalous observations.
                dff_values = dff.values.astype(np.float32)
                autoencoder = MLPRegressor(
                    hidden_layer_sizes=(20, 10, 10, 20),
                    activation='tanh',
                    solver='adam',
                    max_iter=200,
                    random_state=42,
                    verbose=False
                )
                autoencoder.fit(dff_values, dff_values)
                reconstructions = autoencoder.predict(dff_values)
                mse = np.mean(np.power(dff_values - reconstructions, 2), axis=1)
                dfs['autoencoders'] = mse
                threshold = np.percentile(mse, 99)
                dfs['autoencoder_anomaly'] = (mse > threshold).astype(int)

                ########################### 6. Combine All Models ###########################
                dfs['combined_anomalies'] = (dfs['Z_score'] + dfs['lof'] + dfs['isolationforest'] + dfs['autoencoders']) / 4
                dfs['combined_anomalies'] = (np.abs(dfs['combined_anomalies']) > 3).astype(int)

                ########################### Annotate ###########################
                df1['ZScore_anomaly'] = dfs['ZScore_anomaly'].values
                df1['zscore_maxDefRej_anomaly'] = dfs['zscore_maxAmount_anomaly'].values
                df1['lof_anomaly'] = dfs['lof_anomaly'].values
                df1['isolationforest_anomaly'] = dfs['isolationforest_anomaly'].values
                df1['autoencoder_anomaly'] = dfs['autoencoder_anomaly'].values
                df1['combined_anomalies'] = dfs['combined_anomalies'].values

                df1['anomaly_indicator'] = df1[['ZScore_anomaly', 'zscore_maxDefRej_anomaly', 'lof_anomaly', 'isolationforest_anomaly', 'autoencoder_anomaly', 'combined_anomalies']].sum(axis=1) > 0

                df1['a_type'] = 0
                df1.loc[df1.ZScore_anomaly == 1, 'a_type'] = 1
                df1.loc[df1.zscore_maxDefRej_anomaly == 1, 'a_type'] = 2
                df1.loc[df1.lof_anomaly == 1, 'a_type'] = 3
                df1.loc[df1.isolationforest_anomaly == 1, 'a_type'] = 4
                df1.loc[df1.autoencoder_anomaly == 1, 'a_type'] = 5
                df1.loc[df1.combined_anomalies == 1, 'a_type'] = 7

                df1['a_name'] = 'None'
                df1.loc[df1.ZScore_anomaly == 1, 'a_name'] = 'ZScore'
                df1.loc[df1.zscore_maxDefRej_anomaly == 1, 'a_name'] = 'ZScore Max'
                df1.loc[df1.lof_anomaly == 1, 'a_name'] = 'Local Outlier Factor'
                df1.loc[df1.isolationforest_anomaly == 1, 'a_name'] = 'Isolation Forest'
                df1.loc[df1.autoencoder_anomaly == 1, 'a_name'] = 'AutoEncoder'
                df1.loc[df1.combined_anomalies == 1, 'a_name'] = 'Multiple'

                return df1

            @render.plot()
            def scatter_plot():
                df1 = data()
                if df1.empty:
                    return None
                fig, ax = plt.subplots(1)
                if input.only_anomalies() == "a":
                    ax.scatter(df1['Date'], df1['Amount'], c='black')
                    ax.scatter(df1['Date'], df1['Amount'], c=df1['a_type'], s=df1['a_type'] * 20)
                    plt.title("Dataset with Anomalies")
                elif input.only_anomalies() == "b":
                    ax.scatter(df1['Date'], df1['Amount'], c=df1['a_type'], s=df1['a_type'] * 20)
                    plt.title("Only Anomalies")
                return fig

            @render.text
            def detail_data():
                return "Detail Data for Anomalies in Selected Date Range"

            @render.data_frame
            def detail_data_frame():
                df1 = data()
                AllAnomalies_df = df1[df1[['ZScore_anomaly', 'zscore_maxDefRej_anomaly', 'lof_anomaly', 'isolationforest_anomaly', 'autoencoder_anomaly', 'combined_anomalies']].sum(axis=1) > 0]
                AllAnomalies_df = pd.DataFrame(AllAnomalies_df)
                if AllAnomalies_df.empty:
                    return None
                return render.DataGrid(AllAnomalies_df)