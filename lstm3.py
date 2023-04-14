# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Koneksi ke database MySQL
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="pengujian"
)

# Mengambil data dari database MySQL
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM honda")
data = mycursor.fetchall()

# Membuat DataFrame dari data MySQL
df = pd.DataFrame(data, columns=['tahun', 'minat', 'trand', 'penjualan'])

# Mengubah tipe data kolom 'sales' menjadi float
df['penjualan'] = df['penjualan'].astype(float)

# Menormalkan data menggunakan MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['penjualan']])

# Membuat data training dan testing
train_data = scaled_data[:int(len(df)*0.8)]
test_data = scaled_data[int(len(df)*0.8):]

# Membuat fungsi untuk membuat dataset dengan time steps
def create_dataset(data, time_steps=1):
    X_data, Y_data = [], []
    for i in range(len(data)-time_steps):
        X_data.append(data[i:(i+time_steps), 0])
        Y_data.append(data[i+time_steps, 0])
    return np.array(X_data), np.array(Y_data)

# Membuat dataset untuk training dan testing
time_steps = 10 # Contoh: menggunakan 12 bulan sebelumnya untuk memprediksi penjualan bulan berikutnya
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# Reshape data agar sesuai dengan format input LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Membuat model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=2)

# Menggunakan model untuk memprediksi data testing
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Menghitung mean squared error (MSE)
mse = np.mean((predictions - Y_test)**2)
print("MSE:", mse)
