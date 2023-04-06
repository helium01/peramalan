# Impor pustaka yang diperlukan
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import mysql.connector

# Baca data dari file CSV
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="pengujian"
)

# Buat cursor
mycursor = mydb.cursor()

# Eksekusi query untuk mengambil data
mycursor.execute("SELECT * FROM `honda` ORDER BY `honda`.`tahun` ASC")

# Fetch semua data
data = mycursor.fetchall()
df = pd.DataFrame(data, columns=['tahun','minat', 'trand', 'penjualan'])
data=df
# Ubah format tanggal ke dalam format datetime dan set sebagai indeks
data['tahun'] = pd.to_datetime(data['tahun'])
data = data.set_index('tahun')

# Pisahkan data menjadi data latih dan data uji

train_data = data.loc[:-12]
test_data = data.loc[-12:]

# Normalisasi data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Buat data latih dan data uji
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), :])
        Y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
train_X, train_Y = create_dataset(scaled_train_data, look_back)
test_X, test_Y = create_dataset(scaled_test_data, look_back)

# Buat model LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Latih model
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Evaluasi model pada data uji
test_predict = model.predict(test_X)
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])
rmse = np.sqrt(np.mean((test_predict - test_Y)**2))
print('RMSE:', rmse)

# Membuat peramalan
last_three_months = scaled_test_data[-3:, :]
future_data = np.array([last_three_months])
for i in range(12):
    prediction = model.predict(future_data)
    future_data = np.append(future_data[:, 1:, :], prediction, axis=1)

# Konversi hasil peramalan ke skala semula
future_data = scaler.inverse_transform(future_data.reshape(-1, 4))

# Tampilkan hasil peramalan
print('Peramalan penjualan kendaraan untuk 12 bulan mendatang:')
print(future_data[:, 0])
