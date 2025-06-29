import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_prophet_model(
    df,
    freq,
    periods_input,
    use_holidays,
    yearly,
    weekly,
    daily,
    seasonality_mode,
    changepoint_prior_scale,
):
    if use_holidays:
        years = df["ds"].dt.year.unique()
        holiday_dates = []
        for year in years:
            holiday_dates.extend(
                [
                    f"{year}-01-01",
                    f"{year}-01-06",
                    f"{year}-04-25",
                    f"{year}-05-01",
                    f"{year}-06-02",
                    f"{year}-08-15",
                    f"{year}-11-01",
                    f"{year}-12-08",
                    f"{year}-12-25",
                    f"{year}-12-26",
                ]
            )
        holidays = pd.DataFrame(
            {"ds": pd.to_datetime(holiday_dates), "holiday": "festività_italiane"}
        )
        model = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            holidays=holidays,
        )
    else:
        model = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    model.fit(df)
    future = model.make_future_dataframe(periods=periods_input, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def evaluate_forecast(df, forecast):
    df_forecast = forecast[["ds", "yhat"]].set_index("ds")
    df_actual = df.set_index("ds")
    df_combined = df_actual.join(df_forecast, how="left").dropna()

    mae = mean_absolute_error(df_combined["y"], df_combined["yhat"])
    mse = mean_squared_error(df_combined["y"], df_combined["yhat"])
    rmse = mse**0.5
    mape = (
        np.mean(np.abs((df_combined["y"] - df_combined["yhat"]) / df_combined["y"]))
        * 100
    )

    return mae, rmse, mape, df_combined


def plot_forecast(model, forecast):
    return plot_plotly(model, forecast)


def plot_components(model, forecast):
    return plot_components_plotly(model, forecast)


def run_prophet_model(
    df: pd.DataFrame,
    forecast_periods: int = 30,
    freq: str = "D",
    use_holidays: bool = False,
    yearly: bool | str = "auto",
    weekly: bool | str = "auto",
    daily: bool | str = "auto",
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
):
    """Run Prophet forecasting with a Streamlit interface."""
    import streamlit as st

    st.header("Prophet Forecasting")
    st.write(
        f"Orizzonte: {forecast_periods} periodi | Freq: {freq} | Holidays: {use_holidays}"
    )
    st.write("Anteprima dati:")
    st.dataframe(df.head())

    try:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        model, forecast = build_prophet_model(
            df,
            freq=freq,
            periods_input=forecast_periods,
            use_holidays=use_holidays,
            yearly=yearly,
            weekly=weekly,
            daily=daily,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
        )

        st.success("Modello Prophet addestrato.")

        mae, rmse, mape, _ = evaluate_forecast(df, forecast)
        st.subheader("\U0001f4dd Metriche di errore")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        fig_forecast = plot_forecast(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        fig_components = plot_components(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)
    except Exception as e:
        st.error(f"Errore nell'esecuzione del modello: {e}")
