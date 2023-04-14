from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector

# Load data
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="pengujian"
)

# Membaca data dari database
query = "SELECT * FROM `honda` ORDER BY `honda`.`tahun` ASC"
df = pd.read_sql(query, con=mydb)

myresult = df.fetchall()
df = pd.DataFrame(myresult, columns=['tahun', 'penjualan', 'minat', 'trand'])
cols=['tahun']
df.drop(cols,axis=1,inplace=True)
dataset=df
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split data into training and testing sets
train_size = int(len(dataset) * 0.8)
train_data = dataset[0:train_size, :]
test_data = dataset[train_size:len(dataset), :]

# Preprocess data
def preprocess(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), :])
        Y.append(data[i+look_back, 3]) # Predict sales column
    return np.array(X), np.array(Y)

look_back = 3
trainX, trainY = preprocess(train_data, look_back)
testX, testY = preprocess(test_data, look_back)

# Define LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse normalize the predictions
trainPredict = scaler.inverse_transform(np.concatenate((trainX[:, -1, 1:], trainPredict), axis=1))[:, -1]
testPredict = scaler.inverse_transform(np.concatenate((testX[:, -1, 1:], testPredict), axis=1))[:, -1]

# Plot the results
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 3] = trainPredict
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 3] = testPredict
plt.plot(scaler.inverse_transform(dataset)[:, 3])
plt.plot(trainPredictPlot[:, 3])
plt.plot(testPredictPlot[:, 3])
plt.show()