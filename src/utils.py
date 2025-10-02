import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_true, y_pred, pollutant_names=None):
    """
    Calculate comprehensive evaluation metrics for air quality predictions

    Args:
        y_true: True values (array-like)
        y_pred: Predicted values (array-like)  
        pollutant_names: List of pollutant names ['O3', 'NO2']

    Returns:
        Dictionary of metrics
    """
    if pollutant_names is None:
        pollutant_names = ['O3', 'NO2']

    metrics = {}

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    for i, pollutant in enumerate(pollutant_names):
        if i < y_true.shape[1]:
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]

            # Remove NaN values
            mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
            true_vals = true_vals[mask]
            pred_vals = pred_vals[mask]

            if len(true_vals) > 0:
                rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
                r2 = r2_score(true_vals, pred_vals)
                mae = mean_absolute_error(true_vals, pred_vals)
                ria = calculate_ria(true_vals, pred_vals)

                metrics[f'{pollutant.lower()}_rmse'] = rmse
                metrics[f'{pollutant.lower()}_r2'] = r2
                metrics[f'{pollutant.lower()}_mae'] = mae
                metrics[f'{pollutant.lower()}_ria'] = ria

    return metrics

def calculate_ria(y_true, y_pred):
    """
    Calculate Refined Index of Agreement (RIA)

    RIA = 1 - [Σ(Pi - Oi)²] / [Σ(|Pi - O̅| + |Oi - O̅|)²]
    where Pi = predicted, Oi = observed, O̅ = observed mean
    """
    mean_obs = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    ria = 1 - (numerator / denominator)
    return max(0, min(1, ria))  # Clamp between 0 and 1

def create_site_map(site_coords, selected_site=None):
    """
    Create interactive map of Delhi monitoring sites

    Args:
        site_coords: Dictionary of site coordinates
        selected_site: ID of currently selected site

    Returns:
        Plotly figure object
    """
    site_df = pd.DataFrame([
        {
            'Site': f"Site {k}",
            'lat': v['lat'], 
            'lon': v['lon'], 
            'name': v['name'],
            'selected': k == selected_site
        } for k, v in site_coords.items()
    ])

    # Create map
    fig = px.scatter_mapbox(
        site_df, 
        lat="lat", 
        lon="lon", 
        hover_name="Site",
        hover_data=["name"],
        color="selected",
        color_discrete_map={True: "#FF4B4B", False: "#1F77B4"},
        size_max=15,
        zoom=10,
        mapbox_style="open-street-map",
        title="Delhi Air Quality Monitoring Sites"
    )

    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        height=400
    )

    return fig

def create_forecast_plot(pred_df, site_id, selected_date):
    """
    Create comprehensive forecast visualization

    Args:
        pred_df: DataFrame with predictions
        site_id: Site identifier
        selected_date: Date for the forecast

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # O3 predictions
    fig.add_trace(go.Scatter(
        x=pred_df['datetime'],
        y=pred_df['O3_pred'],
        name='O₃ Prediction',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers',
        hovertemplate='<b>O₃</b><br>' +
                      'Time: %{x}<br>' +
                      'Concentration: %{y:.1f} μg/m³<br>' +
                      '<extra></extra>'
    ))

    # NO2 predictions
    fig.add_trace(go.Scatter(
        x=pred_df['datetime'],
        y=pred_df['NO2_pred'],
        name='NO₂ Prediction',
        line=dict(color='#ff7f0e', width=3),
        mode='lines+markers',
        yaxis='y2',
        hovertemplate='<b>NO₂</b><br>' +
                      'Time: %{x}<br>' +
                      'Concentration: %{y:.1f} μg/m³<br>' +
                      '<extra></extra>'
    ))

    # Add forecast baselines if available
    if 'O3_forecast' in pred_df.columns:
        fig.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['O3_forecast'],
            name='O₃ Input Forecast',
            line=dict(color='lightblue', dash='dash', width=2),
            opacity=0.7,
            hovertemplate='<b>O₃ Forecast Input</b><br>' +
                          'Time: %{x}<br>' +
                          'Concentration: %{y:.1f} μg/m³<br>' +
                          '<extra></extra>'
        ))

    if 'NO2_forecast' in pred_df.columns:
        fig.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['NO2_forecast'],
            name='NO₂ Input Forecast',
            line=dict(color='lightsalmon', dash='dash', width=2),
            opacity=0.7,
            yaxis='y2',
            hovertemplate='<b>NO₂ Forecast Input</b><br>' +
                          'Time: %{x}<br>' +
                          'Concentration: %{y:.1f} μg/m³<br>' +
                          '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"24-Hour Air Quality Forecast - Site {site_id} - {selected_date}",
        xaxis=dict(
            title="Time (Hours)",
            tickformat="%H:%M",
            dtick="H1"
        ),
        yaxis=dict(
            title="O₃ Concentration (μg/m³)",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            side="left"
        ),
        yaxis2=dict(
            title="NO₂ Concentration (μg/m³)",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            side="right",
            overlaying="y"
        ),
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    return fig

def format_metrics_display(metrics):
    """
    Format metrics for display in Streamlit

    Args:
        metrics: Dictionary of calculated metrics

    Returns:
        Formatted metrics dictionary
    """
    formatted = {}

    for key, value in metrics.items():
        if 'rmse' in key or 'mae' in key:
            formatted[key] = f"{value:.2f} μg/m³"
        elif 'r2' in key or 'ria' in key:
            formatted[key] = f"{value:.3f}"
        else:
            formatted[key] = f"{value:.2f}"

    return formatted

def validate_data_quality(df, required_cols):
    """
    Validate data quality and completeness

    Args:
        df: DataFrame to validate
        required_cols: List of required columns

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'messages': [],
        'missing_cols': [],
        'missing_data_pct': {}
    }

    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results['valid'] = False
        results['missing_cols'] = missing_cols
        results['messages'].append(f"Missing required columns: {missing_cols}")

    # Check missing data percentage
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        results['missing_data_pct'][col] = missing_pct

        if missing_pct > 50:
            results['messages'].append(f"Column '{col}' has {missing_pct:.1f}% missing data")

    # Check date range
    if all(col in df.columns for col in ['year', 'month', 'day']):
        try:
            dates = pd.to_datetime(df[['year', 'month', 'day']])
            date_range = dates.max() - dates.min()
            results['messages'].append(f"Data spans {date_range.days} days")
        except:
            results['messages'].append("Unable to parse date columns")

    return results

def generate_model_summary(model_type, metrics, data_info):
    """
    Generate a comprehensive model summary

    Args:
        model_type: Type of model used
        metrics: Performance metrics
        data_info: Information about the dataset

    Returns:
        Formatted summary string
    """
    summary = f"""
    ## Model Performance Summary

    **Model Type**: {model_type.upper()}
    **Dataset**: {data_info.get('samples', 'Unknown')} samples
    **Features**: {data_info.get('features', 'Unknown')} features

    ### O₃ Performance
    - RMSE: {metrics.get('o3_rmse', 'N/A'):.2f} μg/m³
    - R²: {metrics.get('o3_r2', 'N/A'):.3f}
    - MAE: {metrics.get('o3_mae', 'N/A'):.2f} μg/m³
    - RIA: {metrics.get('o3_ria', 'N/A'):.3f}

    ### NO₂ Performance  
    - RMSE: {metrics.get('no2_rmse', 'N/A'):.2f} μg/m³
    - R²: {metrics.get('no2_r2', 'N/A'):.3f}
    - MAE: {metrics.get('no2_mae', 'N/A'):.2f} μg/m³
    - RIA: {metrics.get('no2_ria', 'N/A'):.3f}
    """

    return summary
