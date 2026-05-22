import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

st.title("🤖 AI Prediction")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

data = yf.download(stock, period="5y")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data_train = pd.DataFrame(
    data['Close'][0:int(len(data)*0.70)]
)

data_test = pd.DataFrame(
    data['Close'][int(len(data)*0.70):]
)

scaler = MinMaxScaler(feature_range=(0,1))

data_train_scale = scaler.fit_transform(data_train)

model = load_model("stock_prediction_model.keras")

past_100_days = data_train.tail(100)

final_df = pd.concat(
    [past_100_days, data_test],
    ignore_index=True
)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("📈 Original vs Predicted")

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=y_test,
    name='Original Price'
))

fig.add_trace(go.Scatter(
    y=y_predicted.flatten(),
    name='Predicted Price'
))

fig.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

mae = mean_absolute_error(
    y_test,
    y_predicted
)

st.metric(
    "Model MAE",
    round(float(mae),2)
)

# Future Prediction
st.subheader("🔮 Future 7-Day Forecast")

future_days = 7

last_100_days = input_data[-100:]

future_input = last_100_days.reshape(1,100,1)

future_predictions = []

for i in range(future_days):

    next_day = model.predict(future_input)

    future_predictions.append(next_day[0,0])

    next_day_reshaped = next_day.reshape(1,1,1)

    future_input = np.concatenate(
        (future_input[:,1:,:], next_day_reshaped),
        axis=1
    )

future_predictions = np.array(future_predictions)

future_predictions = future_predictions * scale_factor

future_dates = pd.date_range(
    start=data.index[-1],
    periods=future_days + 1
)[1:]

future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions.flatten()
})

st.write(future_df)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['Predicted Price'],
    mode='lines+markers'
))

fig2.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)