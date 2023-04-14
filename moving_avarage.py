import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Lihat 5 data teratas
# print(motor_sales_data.head())
#
# plt.figure(figsize=(10, 5))
# plt.plot(motor_sales_data.index, motor_sales_data['penjualan'])
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.title('Motor Sales Data')
# plt.show()

motor_sales_data['moving_avg'] = motor_sales_data['penjualan'].rolling(window=3).mean()

# Melihat 5 data teratas
# print(motor_sales_data.head())

motor_sales_data['ma_with_interest_trend'] = motor_sales_data['moving_avg'] * motor_sales_data['minat'] * motor_sales_data['trand']

# Melihat 5 data teratas
print(motor_sales_data)
def hasil():
  return motor_sales_data

plt.figure(figsize=(10, 5))
plt.plot(motor_sales_data.index, motor_sales_data['penjualan'], label='Actual Sales')
plt.plot(motor_sales_data.index, motor_sales_data['ma_with_interest_trend'], label='Moving Average with Interest and Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Motor Sales Data with Moving Average, Interest, and Trend')
plt.legend()
plt.show()

# Menghitung nilai MSE (Mean Squared Error)
mse = ((motor_sales_data['penjualan'] - motor_sales_data['ma_with_interest_trend']) ** 2).mean()
print(f"Mean Squared Error (MSE) : {mse:.2f}")