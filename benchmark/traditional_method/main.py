import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt



def prophet_forecast(data, forecast_steps=10):
    """
    Perform Prophet forecasting on each time series in the data matrix.

    Parameters:
    data (np.ndarray): The input data matrix with shape (N, T) where N is the number of variables
                       and T is the number of time points.
    forecast_steps (int): The number of steps to forecast.

    Returns:
    np.ndarray: The forecasted values with shape (N, forecast_steps).
    """
    N, T = data.shape
    forecasted_values = np.zeros((N, forecast_steps))
    
    for i in range(N):
        # Prepare the dataframe for Prophet
        series = data[i, :]
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=T, freq='D'),
            'y': series
        })
        
        # Fit the model
        model = Prophet()
        model.fit(df)
        
        # Create a dataframe for future predictions
        future = model.make_future_dataframe(periods=forecast_steps)
        
        # Predict
        forecast = model.predict(future)
        
        # Extract the forecasted values
        forecasted_values[i, :] = forecast['yhat'].tail(forecast_steps).values
    
    return forecasted_values

def arima_forecast(data, order=(1, 1, 1), forecast_steps=10):
    data = data + 1
    N, T = data.shape
    forecasted_values = np.zeros((N, forecast_steps))
    for i in tqdm(range(N)):
        series = data[i, :]
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        forecasted_values[i, :] = forecast
    return forecasted_values - 1

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prophet')
    parser.add_argument('--mode', type=str, default='count')
    parser.add_argument('--model', type=str, default='ARIMA')
    parser.add_argument('--data_name', type=str, default='r1')
    parser.add_argument('--scale', type=str, default='season')
    parser.add_argument('--seed', type=int, default=0, help='status')
    parser.add_argument('--forecast_steps', type=int, default=2, help='status')
    args = parser.parse_args()
    np.random.seed(args.seed)
    if 'num' != args.mode:
        col = [f"{i}_id" for i in args.data_name.split('-') + ['skill']]
    else:
        col = [f"{i}_id" for i in args.data_name.split('-')]
    data = pd.read_csv(f'../../data/{args.scale}/job_{args.mode}_{args.data_name}.csv').set_index(col).values[:,:-args.forecast_steps]
    N, T = data.shape
    if args.model == 'ARIMA':
        forecasted_values = arima_forecast(data, order=(1, 1, 1), forecast_steps=args.forecast_steps)
    else:
        forecasted_values = prophet_forecast(data, forecast_steps=args.forecast_steps)
    if not os.path.exists(f"outputs/{args.scale}/{args.mode}/{args.forecast_steps}/{args.data_name}/{args.model}"):
        os.makedirs(f"outputs/{args.scale}/{args.mode}/{args.forecast_steps}/{args.data_name}/{args.model}")
    np.save(f'outputs/{args.scale}/{args.mode}/{args.forecast_steps}/{args.data_name}/{args.model}/pred.npy', forecasted_values)