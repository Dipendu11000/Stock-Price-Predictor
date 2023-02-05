import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import streamlit as st

def lstm_model(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    x_train, y_train = [], []
    for i in range(60, data.shape[0]):
        x_train.append(data[i-60:i,0])
        y_train.append(data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=5, verbose=2)
    last_60_days = data[-60:]
    last_60_days = np.array(last_60_days).reshape(-1, 1)
    input_data = scaler.transform(last_60_days)
    input_data = np.reshape(input_data, (1,input_data.shape[0],1))
    prediction = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]


def main():
    st.title("Stock Price Prediction")
    #st.write("Enter the company name (ticker): ")
    company = st.text_input("Enter the company name (ticker): ")
    df = yf.download(tickers=company, period='1mo', interval='5m')
    df_close = df['Close']
    df_close = df_close.reset_index(drop=True)
    lstm_prediction = lstm_model(df_close)
    st.write("Prediction of the stock value in next 5 minutes: ", lstm_prediction)

if __name__ == "__main__":
    main()
