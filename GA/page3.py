import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import numpy as np
from collections import defaultdict

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    try:
        numeric_columns = df.columns.drop('Year')
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype(int)
            df = df.set_index('Year')

        return df
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None

def calculate_accuracy(test, predictions):
    mae = mean_absolute_error(test, predictions)
    accuracy = (1 - mae / test.mean()) * 100
    return accuracy

def calculate_rmse(test, predictions):
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(test, predictions):
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    return mape

def generate_suggestions(predictions):
    suggestions = defaultdict(list)
    mean_value = predictions.mean()
    for year, value in predictions.items():
        if value > mean_value:
            suggestions["High export value. Consider increasing production."].append(year)
        else:
            suggestions["Low export value. Consider improving product quality."].append(year)
    return suggestions

def format_suggestions(suggestions):
    formatted_suggestions = []
    for suggestion, years in suggestions.items():
        if len(years) > 1:
            year_range = f"{years[0]}-{years[-1]}" if years[-1] - years[0] == len(years) - 1 else ", ".join(
                map(str, years))
        else:
            year_range = str(years[0])
        formatted_suggestions.append(f"{year_range}: {suggestion}")
    return formatted_suggestions

def plot_arima_forecast(data, column, p, d, q, steps=10):
    try:
        data_for_forecast = data.copy()
        data_for_forecast[column] = pd.to_numeric(data_for_forecast[column], errors='coerce')
        data_for_forecast = data_for_forecast.dropna(subset=[column])

        historical_data = data_for_forecast[data_for_forecast.index >= 1990]

        model = ARIMA(historical_data[column], order=(p, d, q))
        model_fit = model.fit()

        forecast = model_fit.get_forecast(steps=steps)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Calculate accuracy
        train_size = int(len(historical_data) * 0.8)
        train, test = historical_data.iloc[:train_size], historical_data.iloc[train_size:]
        model_train = ARIMA(train[column], order=(p, d, q))
        model_fit_train = model_train.fit()
        predictions = model_fit_train.forecast(steps=len(test))

        # Create a DataFrame for test values and predictions
        test_years = test.index
        test_df = pd.DataFrame({'Year': test_years, 'Actual': test[column].values, 'Predicted': predictions})

        # Calculate accuracy metrics
        accuracy = calculate_accuracy(test[column], predictions)

        st.write(f"### Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}%")

        if accuracy < 75:
            st.error("The model accuracy is below 75%. A suitable model couldn't be found.")
            return

        last_year = historical_data.index.max()
        forecast_index = pd.date_range(start=f'{last_year + 1}-01-01', periods=steps, freq='Y').year

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[column],
            mode='lines',
            name='Historical Data',
            hovertemplate = '%{y:.2f}'
        ))

        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_values,
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='orange'),
            hovertemplate = '%{y:.2f}'
        ))

        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=conf_int.iloc[:, 0],
            mode='lines',
            name='Lower Confidence Interval',
            line=dict(width=0),
            showlegend=False,
            hoverinfo = 'skip'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=conf_int.iloc[:, 1],
            mode='lines',
            name='Upper Confidence Interval',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 182, 193, 0.3)',
            showlegend=False,
            hoverinfo = 'skip'
        ))

        fig.update_layout(
            title=f'ARIMA Forecast for {column}',
            xaxis_title='Year',
            yaxis_title='Export Value (RM)',
            hovermode='x'
        )

        st.plotly_chart(fig)
     
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def grid_search_arima(data, column):
    p = d = q = range(0, 4)
    pdq_combinations = list(itertools.product(p, d, q))

    best_aic = np.inf
    best_order = None
    best_model = None
    best_accuracy = 0

    for order in pdq_combinations:
        try:
            model = ARIMA(data[column], order=order)
            model_fit = model.fit()
            aic = model_fit.aic

            # Calculate accuracy
            train_size = int(len(data) * 0.8)
            train, test = data.iloc[:train_size], data.iloc[train_size:]
            model_train = ARIMA(train[column], order=order)
            model_fit_train = model_train.fit()
            predictions = model_fit_train.forecast(steps=len(test))
            accuracy = calculate_accuracy(test[column], predictions)

            if aic < best_aic and accuracy >= 75:
                best_aic = aic
                best_order = order
                best_model = model_fit
                best_accuracy = accuracy

        except:
            continue

    return best_order, best_model, best_accuracy

def page3():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&display=swap');
    body {
        font-family: 'EB Garamond', serif;
        font-size: 20px;
    }
    h2 {
        font-family: 'EB Garamond', serif;
        font-size: 26px;
    }
    h3 {
        font-family: 'EB Garamond', serif;
        font-size: 22px;
    }
    div.st-emotion-cache-0.e1f1d6gn0 {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5px 5px 5px 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Export Data Analysis and Forecasting")
    st.write(
        "This section allows user to analyze and forecast export data using ARIMA models. Select the column to forecast and set the forecast parameters.")
    
    file_path = 'C:/Users/ngwaw/Downloads/A232 SQIT3073/GA/MalaysiaExports.csv'

    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        if df is not None:
            st.header("ARIMA Forecast Options")
            column = st.selectbox("Export Category:", df.columns)
            steps = st.slider("Number of Year to be Forecast:", min_value=2, max_value=20, value=10)

            st.write(
                "User Guide:\n"
                 "1. Select the category.\n"
                 "2. Select the number of years to predict.\n"
                 "3. Click the generate button.\n"
            )
                 

            if st.button("Generate Forecast"):
                with st.spinner('Finding the best ARIMA model...'):
                    best_order, best_model, best_accuracy = grid_search_arima(df, column)
                    if best_model is not None and best_accuracy >= 75:
                        st.success(f"Best ARIMA order: {best_order} with accuracy: {best_accuracy:.2f}%")
                        plot_arima_forecast(df, column, best_order[0], best_order[1], best_order[2], steps)
                    else:
                        st.error("Failed to find a suitable ARIMA model with at least 75% accuracy.")

if __name__ == "__main__":
    page3()
