import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.modules.utils.metrics_module import compute_all_metrics

def run_arima_model(df: pd.DataFrame, date_col: str, target_col: str, horizon: int, selected_metrics: list, params: dict, return_metrics=False):
    if not return_metrics:
        st.subheader("ARIMA Forecast")

    p = params.get('p', 1)
    d = params.get('d', 1)
    q = params.get('q', 0)

    try:
        series = df.set_index(date_col)[target_col]
        series.index = pd.to_datetime(series.index)
        series = series.asfreq(pd.infer_freq(series.index), method='ffill')

        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()

        if not return_metrics:
            st.success("Modello ARIMA addestrato con successo.")
            with st.expander("Vedi sommario del modello"):
                st.text(str(model_fit.summary()))

        fitted = model_fit.fittedvalues
        forecast = model_fit.get_forecast(steps=horizon)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Calcolo e visualizzazione metriche
        if not return_metrics:
            st.write("### Evaluation Metrics (su dati storici)")
        metrics_results = compute_all_metrics(series, fitted)
        if not selected_metrics: selected_metrics = ["MAE", "RMSE", "MAPE"]
        
        if not return_metrics:
            cols = st.columns(len(selected_metrics))
            for i, metric in enumerate(selected_metrics):
                value = metrics_results.get(metric)
                if value is not None:
                    format_str = "{:.0f}%" if metric in ["MAPE", "SMAPE"] else "{:.3f}"
                    cols[i].metric(metric, format_str.format(value))

        # Grafico con Plotly
        if not return_metrics:
            st.write("### Grafico Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name='Storico', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=fitted.index, y=fitted, mode='lines', name='Fitted', line=dict(color='#ff7f0e', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='#d62728')))
            fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 0], fill='tonexty', mode='none', fillcolor='rgba(214,39,40,0.2)', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:, 1], fill='tonexty', mode='none', fillcolor='rgba(214,39,40,0.2)', showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

            # Bottone di download
            if st.button("📥 Scarica Forecast in Excel", key="arima_download_btn"):
                forecast_df = pd.DataFrame({'ds': forecast_mean.index, 'yhat': forecast_mean.values, 'yhat_lower': conf_int.iloc[:, 0], 'yhat_upper': conf_int.iloc[:, 1]})
                buffer = io.BytesIO()
                forecast_df.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                st.download_button(
                    label="Download .xlsx",
                    data=buffer,
                    file_name="arima_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Errore durante l'esecuzione del modello ARIMA: {e}")
        if return_metrics:
            return {}
    
    # Restituisce le metriche se richiesto (per il modulo Exploratory)
    if return_metrics:
        return metrics_results if 'metrics_results' in locals() else {}
    
    return None  # Default per mantenere compatibilità
