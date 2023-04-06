import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

X_train = train_data.drop(['penjualan'], axis=1)
y_train = train_data['penjualan']
X_test = test_data.drop(['penjualan'], axis=1)
y_test = test_data['penjualan']

# Buat model regresi linier
model = LinearRegression()

# Latih model
model.fit(X_train, y_train)

# Gunakan model untuk melakukan prediksi pada data uji
predictions = model.predict(X_test)

# Tampilkan hasil prediksi
print(predictions)


print(f"Koefisien model : {model.coef_}")
print(f"Intercept model : {model.intercept_}")

# Visualisasikan hasil prediksi dan data uji
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data['penjualan'].values, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()

