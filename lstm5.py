import datetime
import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Mengambil data dari database MySQL
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

# Membuat fungsi untuk membuat dataset dengan time steps
def create_dataset(data, time_steps=1):
    X_data, Y_data = [], []
    for i in range(len(data)-time_steps):
        X_data.append(data[i:(i+time_steps), 0])
        Y_data.append(data[i+time_steps, 0])
    return np.array(X_data), np.array(Y_data)

# Membuat dataset untuk training dan testing
time_steps = 12 # Contoh: menggunakan 12 bulan sebelumnya untuk memprediksi penjualan bulan berikutnya
data, _ = create_dataset(scaled_data, time_steps)

# Reshape data agar sesuai dengan format input LSTM
data = np.reshape(data, (data.shape[0], data.shape[1], 1))

# Membuat model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data, scaled_data[time_steps:], epochs=100, batch_size=16, verbose=2)

# Menggunakan model untuk memprediksi penjualan kendaraan bermotor selama 3 tahun ke depan
last_data = scaled_data[-time_steps:]
future_predictions = []
for i in range(36): # Memprediksi selama 3 tahun ke depan
    input_data = np.reshape(last_data, (1, time_steps, 1))
    prediction = model.predict(input_data)
    future_predictions.append(prediction[0][0])
    last_data = np.append(last_data[1:], prediction[0][0])

# Invers scaling untuk mendapatkan hasil yang sebenarnya
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Menambahkan nilai tahun ke data prediksi
last_year = df['tahun'].max()
future_years = [str(last_year+datetime.timedelta(i+1)) for i in range(len(future_predictions))]
future_predictions_df = pd.DataFrame(columns=['tahun', 'penjualan'])
future_predictions_df['tahun'] = future_years
future_predictions_df['penjualan'] = float('nan')

for i, prediction in enumerate(future_predictions):
    future_predictions_df.loc[future_predictions_df['tahun']==future_years[i], 'penjualan'] = prediction

# Gabungkan DataFrame df dengan future_predictions_df
df = pd.concat([df, future_predictions_df])

# Cetak DataFrame hasil gabungan
print(df)
