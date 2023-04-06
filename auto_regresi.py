import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
# %matplotlib inline

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

# Konversi data ke dalam DataFrame pandas
motor_sales_data = pd.DataFrame(data, columns=['tahun', 'penjualan', 'minat', 'trand'])

# Ubah kolom 'tanggal' menjadi index
motor_sales_data = motor_sales_data.set_index('tahun')
print(motor_sales_data.head())

plt.figure(figsize=(10, 5))
plt.plot(motor_sales_data.index, motor_sales_data['penjualan'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Motor Sales Data')
plt.show()

train_data = motor_sales_data.iloc[:-12] # menggunakan data sebelum 12 bulan terakhir sebagai data latih
test_data = motor_sales_data.iloc[-12:] # menggunakan 12 bulan terakhir sebagai data uji

# Melihat jumlah data latih dan data uji
print(f"Jumlah data latih : {len(train_data)}")
print(f"Jumlah data uji : {len(test_data)}")

model = AutoReg(train_data['penjualan'], lags=3)

# Latih model
model_fit = model.fit()
# print(model_fit.summary())
# print(model)
# Gunakan model untuk melakukan prediksi pada data uji
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
#
# # Tampilkan hasil prediksi
print(predictions)
#
# Visualisasikan hasil prediksi dan data uji
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data['penjualan'], label='data aktual')
plt.plot(test_data.index, predictions, label='data prediksi')
plt.xlabel('tahun')
plt.ylabel('penjualan')
plt.title('penjualan motor dengan auto regresi')
plt.legend()
plt.show()

# Menghitung nilai MSE (Mean Squared Error)
mse = ((motor_sales_data['penjualan'] - motor_sales_data['ma_with_interest_trend']) ** 2).mean()
print(f"Mean Squared Error (MSE) : {mse:.2f}")