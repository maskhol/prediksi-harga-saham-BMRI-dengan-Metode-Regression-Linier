import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import math
sns.set_style("darkgrid")


def hitung_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def hitung_rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))


def hitung_mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)


st.title("PREDIKSI SAHAM PT. BANK MANDIRI TBK (STUDI KASUS: BURSA EFEK INDONESIA TAHUN 2016-2022)")

st.markdown(
    r"""

	# Manfaat Penelitian
	Manfaat dari penelitian ini adalah :

	1.	Mengetahui hasil prediksi nilai saham perusahaan tahun dari 2016 hingga 2022 yang akan digunakan oleh investor untuk berinvestasi saham di IDX.
	2.	Bagi peneliti selanjutnya, penulis berharap dapat digunakan sebagai referensi untuk penelitian yang akan datang terkait dengan forecasting atau prediksi harga saham.

    """
)

st.sidebar.title("Operasi pada Dataset")
w1 = st.sidebar.checkbox("Tampilkan Dataset", False)
w2 = st.sidebar.checkbox(
    "Menampilkan Setiap Nama Atribut Dan Deskripsinya", False)
plot = st.sidebar.checkbox("Tampilkan Plots", False)
plothist = st.sidebar.checkbox("Tampilkan Hist Plots", False)
linechart = st.sidebar.checkbox("Tampilkan Diagram Garis", False)
trainmodel = st.sidebar.checkbox("Melatih Model", False)
prediksi = st.sidebar.checkbox("Tampikan Prediksi", False)


def read_data():
    return pd.read_csv("./BMRI.2016-2022.csv")[["Date", "Open", "High", "Low", "Close"]]


df = read_data()

if w1:
    st.title("Dataset BMRI")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(df)

if w2:
    st.markdown("""

        # Nama Kolom / atribut dan Keterangan

        Date  : Tanggal jalannya perdagangan.

        Open  : Harga pembukaan pada hari tersebut.

        High  : Harga tertinggi pada hari tersebut.

        Low   : Harga terendah pada hari tersebut.

        Close : Harga penutupan pada hari tersebut.
    
""")

if plothist:
    st.subheader("Distribusi setiap feature")
    options = ("Date", "Open", "High", "Low", "Close")
    sel_cols = st.selectbox("Pilih feature", options, 1)
    st.write(sel_cols)
    fig = go.Histogram(x=df[sel_cols], nbinsx=50)
    st.plotly_chart([fig])

if plot:
    st.subheader("Korelasi antara close dan independent variabel")
    options = ("Date", "Open", "High", "Low", "Close")
    w7 = st.selectbox("Pilih kolom", options, 1)
    st.write(w7)
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.scatter(df[w7], df["Close"])
    plt.xlabel(w7)
    plt.ylabel("Close")
    st.pyplot(fig)

if linechart:
    st.subheader("Diagram garis untuk semua feature")
    cols = ["Open", "High", "Low", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("Date")
    df[cols]
    st.line_chart(df)

if trainmodel:
    st.header("Pemodelan")
    data = df

    # Mengonversi kolom 'date' menjadi tipe data datetime
    data['DateTime'] = pd.to_datetime(data['Date'])

    # Memisahkan data menjadi train dan test berdasarkan tanggal
    train_start_date = pd.to_datetime('2016-01-04')
    train_end_date = pd.to_datetime('2021-12-30')
    test_start_date = pd.to_datetime('2022-01-03')
    test_end_date = pd.to_datetime('2022-12-30')

    # Data Train
    train_data = data.loc[(data['DateTime'] >= train_start_date)
                          & (data['DateTime'] <= train_end_date)]
    X_train = train_data[['Open', 'High', 'Low']].values  # Fitur
    y_train = train_data['Close']  # Target

    # Data Test
    test_data = data.loc[(data['DateTime'] >= test_start_date)
                         & (data['DateTime'] <= test_end_date)]
    X_test = test_data[['Open', 'High', 'Low']].values  # Fitur
    y_test = test_data['Close']  # Target

    lrgr = LinearRegression()
    lrgr.fit(X_train, y_train)
    pred = lrgr.predict(X_test)

    mse = hitung_mse(y_test, pred)
    rmse = hitung_rmse(y_test, pred)
    mape = hitung_mape(y_test, pred)

    mse_li = []
    rmse_li = []
    mape_li = []

    mse_li.append(mse)
    rmse_li.append(rmse)
    mape_li.append(mape)

    summary = pd.DataFrame(
        columns=["MSE", "RMSE", "MAPE"])

    summary["MSE"] = mse_li
    summary["RMSE"] = rmse_li
    summary["MAPE"] = mape_li

    summary["MSE"] = summary["MSE"].round(2)
    summary["MAPE"] = summary["MAPE"].round(2).map("{:.2f}%".format)

    st.write(summary)
    st.success('Model berhasil dilatih')

if prediksi:
    st.subheader("Tampikan Prediksi")
    # Membaca data dari file CSV atau sumber data lainnya
    data = df

    # Mengonversi kolom 'date' menjadi tipe data datetime
    data['DateTime'] = pd.to_datetime(data['Date'])

    # Memisahkan data menjadi train dan test berdasarkan tanggal
    train_start_date = pd.to_datetime('2016-01-04')
    train_end_date = pd.to_datetime('2021-12-30')
    test_start_date = pd.to_datetime('2022-01-03')
    test_end_date = pd.to_datetime('2022-12-30')

    # Data Train
    train_data = data.loc[(data['DateTime'] >= train_start_date)
                          & (data['DateTime'] <= train_end_date)]
    X_train = train_data[['Open', 'High', 'Low']].values  # Fitur
    y_train = train_data['Close']  # Target

    # Data Test
    test_data = data.loc[(data['DateTime'] >= test_start_date)
                         & (data['DateTime'] <= test_end_date)]
    X_test = test_data[['Open', 'High', 'Low']].values  # Fitur
    y_test = test_data['Close']  # Target

    lrgr = LinearRegression()
    lrgr.fit(X_train, y_train)
    pred = lrgr.predict(X_test)

    # menampilkan hasil pred dalam plot dengan x = tanggal dan y = prediksi
    fig = plt.figure()
    plt.plot(test_data['Date'].values, pred, label="Prediksi")
    plt.plot(test_data['Date'].values, y_test.values, label="Aktual")
    plt.legend()
    plt.xlabel("Tanggal")
    plt.ylabel("Prediksi")
    plt.show()
    st.plotly_chart(fig)
    df_pred = pd.DataFrame(columns=["Tanggal", "Aktual", "Prediksi"])
    df_pred["Tanggal"] = test_data['Date']
    df_pred["Aktual"] = y_test.values
    df_pred["Prediksi"] = pred.round(1)
    html = df_pred.to_html(index=False)
    st.markdown(html, unsafe_allow_html=True)
