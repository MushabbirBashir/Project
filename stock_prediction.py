# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
from datetime import datetime
from tkinter import Scale
from turtledemo.sorting_animate import start_qsort

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import mplfinance as mplf
import requests
from bs4 import BeautifulSoup
from keras.src.activations import linear
from pandas.core.interchange.dataframe_protocol import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.losses import Huber
from tensorflow.python.keras.saving.saved_model_experimental import sequential
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from yahoo_fin import stock_info as si
from collections import deque


#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'TSLA'

TRAIN_START = '2021-01-01'     # Start date to read
TRAIN_END = '2023-12-01'       # End date to read




# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo


import yfinance as yf

# Get the data for the stock AAPL
#data = si.get_data(COMPANY, start_date=TRAIN_START, end_date=TRAIN_END)

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
# PRICE_VALUE = "Close"
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# # Note that, by default, feature_range=(0, 1). Thus, if you want a different
# # feature_range (min,max) then you'll need to specify it here
# scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))
# # Flatten and normalise the data
# # First, we reshape a 1D array(n) to 2D array(n,1)
# # We have to do that because sklearn.preprocessing.fit_transform()
# # requires a 2D array
# # Here n == len(scaled_data)
# # Then, we scale the whole array to the range (0,1)
# # The parameter -1 allows (np.)reshape to figure out the array size n automatically
# # values.reshape(-1, 1)
# # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# # When reshaping an array, the new shape must contain the same number of elements
# # as the old shape, meaning the products of the two shapes' dimensions must be equal.
# # When using a -1, the dimension corresponding to the -1 will be the product of
# # the dimensions of the original array divided by the product of the dimensions
# # given to reshape so as to maintain the same number of elements.
#
# # Number of days to look back to base the prediction
# PREDICTION_DAYS = 60 # Original
#
# # To store the training data
# x_train = []
# y_train = []
#
# scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# # Prepare the data
# for x in range(PREDICTION_DAYS, len(scaled_data)):
#     x_train.append(scaled_data[x-PREDICTION_DAYS:x])
#     y_train.append(scaled_data[x])
#
# # Convert them into an array
# x_train, y_train = np.array(x_train), np.array(y_train)
# # Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# # and q = PREDICTION_DAYS; while y_train is a 1D array(p)
#
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# # We now reshape x_train into a 3D array(p, q, 1); Note that x_train
# # is an array of p inputs with each input being a 2D array
#
# #------------------------------------------------------------------------------
# # Build the Model
# ## TO DO:
# # 1) Check if data has been built before.
# # If so, load the saved data
# # If not, save the data into a directory
# # 2) Change the model to increase accuracy?
# #------------------------------------------------------------------------------
# model = Sequential() # Basic neural network
# # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# # for some useful examples
#
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# # This is our first hidden layer which also spcifies an input layer.
# # That's why we specify the input shape for this layer;
# # i.e. the format of each training example
# # The above would be equivalent to the following two lines of code:
# # model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# # model.add(LSTM(units=50, return_sequences=True))
# # For som eadvances explanation of return_sequences:
# # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# # As explained there, for a stacked LSTM, you must set return_sequences=True
# # when stacking LSTM layers so that the next LSTM layer has a
# # three-dimensional sequence input.
#
# # Finally, units specifies the number of nodes in this layer.
# # This is one of the parameters you want to play with to see what number
# # of units will give you better prediction quality (for your problem)
#
# model.add(Dropout(0.2))
# # The Dropout layer randomly sets input units to 0 with a frequency of
# # rate (= 0.2 above) at each step during training time, which helps
# # prevent overfitting (one of the major problems of ML).
#
# model.add(LSTM(units=50, return_sequences=True))
# # More on Stacked LSTM:
# # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
#
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
#
# model.add(Dense(units=1))
# # Prediction of the next closing value of the stock price
#
# # We compile the model by specify the parameters for the model
# # See lecture Week 6 (COS30018)
# model.compile(optimizer='adam', loss='mean_squared_error')
# # The optimizer and loss are two important parameters when building an
# # ANN model. Choosing a different optimizer/loss can affect the prediction
# # quality significantly. You should try other settings to learn; e.g.
#
# # optimizer='rmsprop'/'sgd'/'adadelta'/...
# # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...
#
# # Now we are going to train this model with our training data
# # (x_train, y_train)
# model.fit(x_train, y_train, epochs=25, batch_size=32)
# # Other parameters to consider: How many rounds(epochs) are we going to
# # train our model? Typically, the more the better, but be careful about
# # overfitting!
# # What about batch_size? Well, again, please refer to
# # Lecture Week 6 (COS30018): If you update your model for each and every
# # input sample, then there are potentially 2 issues: 1. If you training
# # data is very big (billions of input samples) then it will take VERY long;
# # 2. Each and every input can immediately makes changes to your model
# # (a souce of overfitting). Thus, we do this in batches: We'll look at
# # the aggreated errors/losses from a batch of, say, 32 input samples
# # and update our model based on this aggregated loss.
#
# # TO DO:
# # Save the model and reload it
# # Sometimes, it takes a lot of effort to train your model (again, look at
# # a training data with billions of input samples). Thus, after spending so
# # much computing power to train your model, you may want to save it so that
# # in the future, when you want to make the prediction, you only need to load
# # your pre-trained model and run it on the new input for which the prediction
# # need to be made.
#
# #------------------------------------------------------------------------------
# # Test the model accuracy on existing data
# #------------------------------------------------------------------------------
# # Load the test data
# TEST_START = '2023-08-02'
# TEST_END = '2024-07-02'
#
# # test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)
#
# test_data = yf.download(COMPANY,TEST_START,TEST_END)
#
#
# # The above bug is the reason for the following line of code
# # test_data = test_data[1:]
#
# actual_prices = test_data[PRICE_VALUE].values
#
# total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)
#
# model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# # We need to do the above because to predict the closing price of the fisrt
# # PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the
# # data from the training period
#
# model_inputs = model_inputs.reshape(-1, 1)
# # TO DO: Explain the above line
#
# model_inputs = scaler.transform(model_inputs)
# # We again normalize our closing price data to fit them into the range (0,1)
# # using the same scaler used above
# # However, there may be a problem: scaler was computed on the basis of
# # the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# # but there may be a lower/higher price during the test period
# # [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# # greater than one)
# # We'll call this ISSUE #2
#
# # TO DO: Generally, there is a better way to process the data so that we
# # can use part of it for training and the rest for testing. You need to
# # implement such a way
#
# #------------------------------------------------------------------------------
# # Make predictions on test data
# #------------------------------------------------------------------------------
# x_test = []
# for x in range(PREDICTION_DAYS, len(model_inputs)):
#     x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
#
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# # TO DO: Explain the above 5 lines
#
# predicted_prices = model.predict(x_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)
# # Clearly, as we transform our data into the normalized range (0,1),
# # we now need to reverse this transformation
# #------------------------------------------------------------------------------
# # Plot the test predictions
# ## To do:
# # 1) Candle stick charts
# # 2) Chart showing High & Lows of the day
# # 3) Show chart of next few days (predicted)
# #------------------------------------------------------------------------------
#
# plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
# plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
# plt.title(f"{COMPANY} Share Price")
# plt.xlabel("Time")
# plt.ylabel(f"{COMPANY} Share Price")
# plt.legend()
# plt.show()
#
# #------------------------------------------------------------------------------
# # Predict next day
# #------------------------------------------------------------------------------
#
#
# real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#
# prediction = model.predict(real_data)
# prediction = scaler.inverse_transform(prediction)
# print(f"Prediction: {prediction}")
#
# # A few concluding remarks here:
# # 1. The predictor is quite bad, especially if you look at the next day
# # prediction, it missed the actual price by about 10%-13%
# # Can you find the reason?
# # 2. The code base at
# # https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# # gives a much better prediction. Even though on the surface, it didn't seem
# # to be a big difference (both use Stacked LSTM)
# # Again, can you explain it?
# # A more advanced and quite different technique use CNN to analyse the images
# # of the stock price changes to detect some patterns with the trend of
# # the stock price:
# # https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# # Can you combine these different techniques for a better prediction??)



def shuffle_in_unison(a, b):
    """Shuffles two arrays in the same way, ensuring corresponding elements align"""

    state = np.random.get_state() #stores the random state for future use
    np.random.shuffle(a)
    np.random.set_state(state) #sets the state to previously used seed to maintain correspondence
    bp.random.shuffle(b)


def load_data(ticker, start_date=None, end_date=None, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True, test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], save_locally=False, load_locally=False, local_path='data.csv'):
   

    """
    Loads data from Yahoo Finance source, scaling, shuffling, normalizing, splitting, handling NaNs, and saving/loading locally.
    
    Parameters:
        ticker (str/pd.DataFrame): The ticker you want to load, e.g., 'AAPL', 'TSLA', or a DataFrame with data.
        start_date (str): Start date for fetching data (format: 'YYYY-MM-DD').
        end_date (str): End date for fetching data (format: 'YYYY-MM-DD').
        n_steps (int): Historical sequence length (i.e., window size) for prediction. Default is 50.
        scale (bool): Whether to scale prices from 0 to 1. Default is True.
        shuffle (bool): Whether to shuffle the dataset (training & testing). Default is True.
        lookup_step (int): Future lookup step to predict. Default is 1 (e.g., next day).
        split_by_date (bool): Whether to split the dataset by date. Default is True. If False, splits randomly.
        test_size (float): Ratio for test data. Default is 0.2 (20% testing data).
        feature_columns (list): List of features to use for the model. Default is all Yahoo Finance features.
        save_locally (bool): Option to save data locally for future use. Default is False.
        load_locally (bool): Option to load data locally if it exists. Default is False.
        local_path (str): Path to save or load data locally. Default is 'data.csv'.

    Returns:
        dict: A dictionary containing the processed dataset and additional information.
    """
    # Check if local data exists and load it if load_locally is True
    if load_locally and os.path.exists(local_path):
        df = pd.read_csv(local_path, index_col = 0, parse_dates = True)
    else:
        # Load data from Yahoo Finance or from a dataframe if load_locally is False
        if isinstance(ticker, str):
            df = si.get_data(ticker, start_date = start_date, end_date = end_date)
            #allows specification of start date and end date for whole dataset as inputs
        elif isinstance(ticker, pd.DataFrame):
            df = ticker
        else:
            raise TypeError("ticker must be either a str or a pd.dataFrame instance.")

        # Save data locally if requested
        if save_locally:
         df.to_csv(local_path)

    # Ensure all feature columns are present in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the DataFrame."



    # Prepare the result dictionary
    result = {}

    result['feature_columns'] = feature_columns
    #Add a copy of the original dataframe
    result['df'] = df.copy()

    # Add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    # Scaling the data if scale is True (default)
    if scale:
        column_scaler = {}
        # Scale the data
        for column in feature_columns:
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis = 1))
            column_scaler[column] = scaler
        result['column_scaler'] = column_scaler

    targets = []
    for i in range(len(df) - lookup_step):
        target_seq = df['adjclose'].values[i + 1: i + 1 + lookup_step]
        if len(target_seq) == lookup_step:
            targets.append(target_seq)

    targets_array = np.array(targets, dtype=np.float32)
    df = df.iloc[:len(targets_array)]  # Trim the DataFrame to the length of `targets_array`
    df['future'] = list(targets_array) # Assign target values

    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # Drop NaNs
    df.dropna(inplace = True)

    # Handle sequences for training/testing
    sequence_data = []
    sequences = deque(maxlen = n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values , df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    # Prepare X's and y's for model
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Spliting the dataset
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # If split_by_date is False
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size = test_size, shuffle = shuffle)

    # Extract dates for testing set
    dates = result["X_test"][:, -1, -1]
    result["test_df"] = result["df"].loc[dates]
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep = 'first')]

    # Remove dates from training/testing sets and convert them to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

def create_dl_model(sequence_length : int, n_features : int, units : int = 256, cell : tf.keras.layers.Layer = LSTM, n_layers : int = 2, dropout : float = 0.3, loss : str = "mean_squared_error", metrics : list[str] = ["mean_absolute_error"], optimizer : str = "adam", bidirectional : bool = False, prediction_steps : int = 1):
    """
    Creates and compiles a deep learning model for time series prediction using LSTM or other RNN cells.

    Parameters:
        sequence_length (int): Length of the input sequences (i.e., the number of time steps).
        n_features (int): Number of features in each time step of the input data.
        units (int): Number of units in each LSTM or RNN cell. Default is 256.
        cell (tf.keras.layers.Layer): The type of RNN cell to use, e.g., LSTM, GRU. Default is LSTM.
        n_layers (int): Number of recurrent layers in the model. Default is 2.
        dropout (float): Dropout rate for regularization. Default is 0.3.
        loss (str): Loss function for training the model. Default is "mean_squared_error".
        metrics (list[str]): List of metrics to evaluate during training. Default is ["mean_absolute_error"].
        optimizer (str): Optimizer for training the model. Default is "adam".
        bidirectional (bool): Whether to use bidirectional RNN layers. Default is False.
        prediction_steps (int): Number of days into the future to predict the closing price for.

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.
    """

    model = Sequential()

    # Add recurrent layers
    for i in range(n_layers):
        if i == 0:
            # First layer with input shape defined
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True, input_shape = (sequence_length, n_features))))
            else:
                model.add(cell(units, return_sequences = True, input_shape = (sequence_length, n_features)))
        elif i == n_layers - 1:
            # Last layer without returning sequences
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = False)))
            else:
                model.add(cell(units, return_sequences = False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences = True)))
            else:
                model.add(cell(units, return_sequences = True))

        # Add dropout after each layer
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(prediction_steps, activation = "linear"))

    # Compile the model
    model.compile(loss = loss, metrics = metrics, optimizer = optimizer)

    return model


def test_dl_model():

    training_data = load_data(COMPANY, start_date=TRAIN_START, end_date=TRAIN_END, split_by_date=False, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], n_steps=100, lookup_step=30)


    # Experiment 1:
    lstm_model = create_dl_model(
        sequence_length=100,
        n_features=5,
        units=128,
        cell=LSTM,
        n_layers=5,
        dropout=0.3,
        prediction_steps=30

    )

    test_trained_dl(training_data = training_data, model = lstm_model, epochs = 5, batch_size= 32)


def test_trained_dl(training_data : dict, model : tf.keras.Model, epochs : int, batch_size : int):
    model.fit(training_data["X_train"], training_data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(training_data["X_test"], training_data["y_test"]))

    last_sequence = training_data["X_test"][-1]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predictions = model.predict(last_sequence)
    predictions = predictions.flatten()
    scaler = training_data['column_scaler']['adjclose']

    feature_columns = training_data.get('feature_columns', [])
    feature_name = 'adjclose'
    assert feature_name in feature_columns, f"Feature {feature_name} not found"

    feature_index = feature_columns.index(feature_name)

    predictions = scaler.inverse_transform(predictions.reshape(-1,1)).flatten()

    actual = training_data['X_test'][:, -1, feature_index][:7]
    actual = scaler.inverse_transform(actual.reshape(-1,1)).flatten()
    print(f"Prediction : {predictions}")
    # print(actual)


    # plot_candlestick_chart(training_data, actual=actual, predicted=predictions)

    return predictions






def plot_candlestick_chart(data : DataFrame, actual: np.ndarray = None, predicted: np.ndarray = None, num_of_days : int = 1, prediction_only : bool = True):
    """
    Function to plot a candlestick chart for stock market financial data.

    Parameters:
    - data (DataFrame): The input stock market data containing 'Open', 'High', 'Low', 'Close', and 'Volume' columns.
    - num_of_days (int): Number of trading days each candlestick represents. Default is 1.
    """
    # Extract DataFrame from the dictionary
    if isinstance(data, dict) and 'test_df' in data:
        data = data['test_df']
    else:
        raise ValueError("Expected 'data' to be a dictionary containing a DataFrame under the key 'test_df'.")

    # Ensure the data index is in datetime format for accurate time series plotting
    data.index = pd.to_datetime(data.index)

    # If num_of_days is greater than 1, resample the data to aggregate over the specified number of days
    if num_of_days > 1:
        data = data.resample(f'{num_of_days}D').agg({
            'open': 'first',  # Take the first opening price in the resampled period
            'high': 'max',  # Take the maximum high price in the resampled period
            'low': 'min',  # Take the minimum low price in the resampled period
            'close': 'last',  # Take the last closing price in the resampled period
            'volume': 'sum'  # Sum the volumes in the resampled period
        }).dropna()  # Drop any periods with NaNs after resampling

    if not prediction_only:
        # Plot the candlestick chart using mplfinance
        mplf.plot(data, type='candle', style='yahoo', title='Candlestick Chart', ylabel='Price')

    if actual is not None or predicted is not None:
        # Plotting actual vs predicted over the last 7 days
        plt.figure(figsize=(10, 6))
        days = np.arange(1, len(actual)+1)
        plt.plot(days, actual, label='Actual', color='blue', marker='o')
        plt.plot(days, predicted, label='Predicted', color='red', linestyle='--', marker='o')
        plt.title('ARIMA Model - 7 Days Prediction vs Actual')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


# Example usage:
# plot_candlestick_chart(data, num_of_days = 50)

def plot_boxplot_chart(data : DataFrame, window : int = 30, limit : int = 90):
    """
    Function to plot a boxplot chart for stock market financial data using a moving window.

    Parameters:
    - data (DataFrame): The input stock market data containing 'Close' column.
    - window (int): The size of the moving window (in days) for which to compute the boxplot data. Default is 30.
    - limit (int): The number of days to represnt in boxplots, ending in the latest available day. Default is 90.
    """

    # Create a list of rolling window data for the 'close' prices
    rolling_data = [data['close'][i: i + window] for i in range(len(data) - window + 1)]

    # Only take the last 'limit' number of rolling windows for plotting
    rolling_data = rolling_data[-limit:]

    # Initialize the plot
    plt.figure(figsize=(18, 8))


    # Create the boxplot for the rolling windows
    plt.boxplot(rolling_data)

    # Set the plot title and labels
    plt.title(f'Boxplot of Stock Prices Over a {window}-Day Moving Window (for last {limit} days)')
    plt.xlabel('Window Number')  # X-axis represents each rolling window
    plt.ylabel('Stock Price')  # Y-axis represents the distribution of stock prices in each window

    # Display the plot
    plt.show()

# Example usage:
# plot_boxplot_chart(data, window=30)


#test_dl_model()

def create_sarima_model(data: dict, feature_name='adjclose', start_p=1, start_q=1, max_p=3, max_q=3, m=1, seasonal=False):
    """
    Creates and trains a SARIMA model for time series prediction.

    Parameters:
        data (dict): Training data containing input features and target labels.
        feature_name (str): The feature name to train the SARIMA model on, e.g., 'adjclose'. Default is 'adjclose'.
        start_p (int): Starting value of the autoregressive (AR) order. Default is 1.
        start_q (int): Starting value of the moving average (MA) order. Default is 1.
        max_p (int): Maximum value for the autoregressive (AR) order. Default is 3.
        max_q (int): Maximum value for the moving average (MA) order. Default is 3.
        m (int): Seasonal period, e.g., 12 for monthly data. Default is 1.
        seasonal (bool): Whether to include seasonal components. Default is False.

    Returns:
        model: The fitted SARIMA model.
    """
    feature_columns = data.get('feature_columns', [])
    assert feature_name in feature_columns, f"Feature {feature_name} not found"

    feature_index = feature_columns.index(feature_name)

    # Extract the specific feature from the training data for model training
    Xtrain = data['X_train']
    training_series = Xtrain[:, -1, feature_index]

    # Create and train the SARIMA model using auto_arima to find optimal parameters
    model = pm.auto_arima(
        training_series,
        start_p=start_p,
        start_q=start_q,
        max_p=max_p,
        max_q=max_q,
        m=m,
        d=None,
        seasonal=seasonal,
        trace=True,
        error_action='ignore',
        suppress_warning=True,
        stepwise=True
    )
    return model


def test_sarima_model():
    # Load training data
    training_data = load_data(
        COMPANY,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        split_by_date=False,
        feature_columns=['adjclose']
    )

    # Create a SARIMA model with specified parameters
    arima_model = create_sarima_model(training_data, feature_name='adjclose', seasonal=True, m=4)
    print(arima_model.summary())

    feature_columns = training_data.get('feature_columns', [])
    assert 'adjclose' in feature_columns, "Feature not found"
    feature_index = feature_columns.index('adjclose')

    # Use trained model to predict on test set
    test_series = training_data['X_test'][:, -1, feature_index]
    predicted_7days = arima_model.predict(n_periods=30)
    actual_7days = test_series[:30]

    # Uncomment the line below to plot actual vs predicted values using candlestick chart
    # plot_candlestick_chart(training_data, actual=actual_7days, predicted=predicted_7days)


def ensemble_prediction(use_LSTM=True, use_SARIMA=True):
    """
    Creates an ensemble model combining predictions from LSTM and SARIMA models.

    Parameters:
        use_LSTM (bool): Whether to include the LSTM model in the ensemble. Default is True.
        use_SARIMA (bool): Whether to include the SARIMA model in the ensemble. Default is True.

    Returns:
        None: Prints the RMSE of the ensemble model on the test set.
    """
    combined_prediction = 0
    # Load training data with specified features
    training_data = load_data(
        COMPANY,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        split_by_date=False,
        feature_columns=['adjclose', 'volume', 'open', 'high', 'low'],
        n_steps=100,
        lookup_step=7
    )

    lstm_model = None
    lstm_prediction = None

    # Train LSTM model if use_LSTM is True
    if use_LSTM:
        lstm_model = create_dl_model(
            sequence_length=100,
            n_features=5,
            units=128,
            cell=LSTM,
            n_layers=5,
            dropout=0.3,
            prediction_steps=7
        )
        lstm_prediction = test_trained_dl(training_data=training_data, model=lstm_model, epochs=20, batch_size=64)
        combined_prediction += lstm_prediction


    sarima_model = None
    sarima_prediction = None

    # Train SARIMA model if use_SARIMA is True
    if use_SARIMA:
        sarima_model = create_sarima_model(data=training_data, feature_name='adjclose', m=12, seasonal=True)
        sarima_prediction = sarima_model.predict(n_periods=7)
        combined_prediction += sarima_prediction

    if use_SARIMA and use_LSTM:
        # Combine predictions from both models if available
        combined_predictions = combined_prediction / 2
    else:
        combined_predictions = combined_prediction

    # Reshape y_test to match the shape of combined predictions
    y_test = training_data['y_test']
    y_test = y_test.reshape(-1, 1).flatten()
    y_test = y_test[-7:]

    # Calculate and print RMSE for the ensemble model
    rmse = np.sqrt(mean_squared_error(y_test, combined_predictions))
    print(f'Ensemble Model RMSE: {rmse}')


# Run the ensemble prediction function
# ensemble_prediction(use_SARIMA=False)



