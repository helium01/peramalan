import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Koneksi ke database MySQL
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="pengujian"
)

# Membaca data dari database
query = "SELECT * FROM `honda` ORDER BY `honda`.`tahun` ASC"
df = pd.read_sql(query, con=mydb)

# Mengubah data menjadi format waktu
df['tahun'] = pd.to_datetime(df['tahun'])
df.set_index('tahun', inplace=True)

# Memperoleh nilai maksimum dan minimum untuk normalisasi data
scaler = MinMaxScaler()
scaler.fit(df)
data = scaler.transform(df)
training_data_len = int(np.ceil(len(data) * .8))

# Membagi data menjadi data pelatihan dan data validasi
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = data[0:train_size,:], data[train_size:len(data),:]

# Fungsi untuk membuat dataset yang dapat digunakan oleh LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# Mengubah data menjadi format yang dapat digunakan oleh LSTM
look_back = 3
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

# Membangun model LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 3)))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='adam')

# Melatih model
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Menguji dan mengevaluasi model
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Training Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Testing Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))
print(test_data)
# Melakukan peramalan
future_data = data[-10:]
future_data_scaled = scaler.transform(future_data)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

train_predictions = scaler.inverse_transform(trainPredict)
# Y_train = scaler.inverse_transform([trainY])
test_predictions = scaler.inverse_transform(testPredict)
# Y_test = scaler.inverse_transform([testY])


