import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mysql.connector

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
test_data = motor_sales_data.iloc[-12:]

# Tentukan model SARIMA
model = SARIMAX(train_data['penjualan'],
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                trend='c')

results = model.fit()
print(results.summary())

# Lakukan prediksi pada data uji
predictions = results.predict(start='2023-01-01', end='2023-12-01')
print(predictions)
# Visualisasikan hasil prediksi dan data uji
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data['penjualan'].values, label='Actual')
plt.plot(predictions.index, predictions.values, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('SARIMA Prediction')
plt.legend()
plt.show()

# Evaluasi model dengan menggunakan mean squared error (MSE)
mse = ((predictions - test_data['penjualan'])**2).mean()
print(f"Mean Squared Error : {mse}")