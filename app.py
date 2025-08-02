#
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import datetime
import joblib
import pytz
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from utils import (
    load_scaler,
    load_dataset,
    feature_engineering,
    model_predict,
    predict_harga_dari_logret,
)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.dates as mdates

# Load .env
load_dotenv()


def get_api_key(key_name):
    try:
        return st.secrets.get(key_name)
    except Exception:
        return os.getenv(key_name)


# Zona waktu Indonesia
wib = pytz.timezone("Asia/Jakarta")
waktu_update = datetime.datetime.now(wib)

# Path setup
MODEL_PATH = "models/model.h5"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "bitcoindaily.csv"

# Load model dan scaler
scaler = load_scaler(SCALER_PATH)
model = load_model(MODEL_PATH, compile=False)

# Load dan proses dataset
df = load_dataset(DATASET_PATH)
df_fe = feature_engineering(df)

# Refresh otomatis setiap 5 menit
st_autorefresh(interval=300000, key="auto_refresh")

# Setup halaman
st.set_page_config(page_title="Prediksi Harga BTC", page_icon="💰")

# Navigasi
st.sidebar.markdown(
    "<h2 style='text-align: left;'>Navigasi</h2>", unsafe_allow_html=True
)
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Tentang", "Prediksi Harga"])


@st.cache_data(ttl=600)
def get_histori_btc(days=30):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days}
    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.resample("D").mean()  # Agregasi harian
        df.dropna(inplace=True)

        return df
    except Exception as e:
        print("[Histori BTC Error]", e)
        return pd.DataFrame()


# Ambil harga BTC dari API
@st.cache_data(ttl=600)  # Cache 10 menit
def get_btc_api(api_key: str):
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        harga = data["bitcoin"]["usd"]
        waktu = datetime.datetime.now(wib)
        return harga, waktu, "CoinGecko"
    except Exception as e:
        print(f"[CoinGecko Error] {e}")

    # Fallback ke Twelve Data
    try:
        api_key = st.secrets.get("TWELVE_API_KEY", None)
        url = f"https://api.twelvedata.com/price?symbol=BTC/USD&apikey={api_key}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        if "price" in data:
            harga = float(data["price"])
            waktu = datetime.datetime.now(wib)
            return harga, waktu, "Twelve Data"
        else:
            print("[Twelve Data] Response tidak mengandung 'price'")
    except Exception as e:
        print(f"[Twelve Data Error] {e}")

    return None, None, "Unknown"


# Fungsi bantu untuk memformat tanggal ke Bahasa Indonesia
def format_tanggal_indonesia(dt):
    bulan_indonesia = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember",
    }
    return f"{dt.day:02d} {bulan_indonesia[dt.month]} {dt.year}"


# ===========================
#        BERANDA
# ===========================
if page == "Beranda":
    st.title("Aplikasi Prediksi Penutupan Harga Bitcoin")

    st.image("assets/logo.jpg", width=1000)  # sesuaikan path dan ukuran

    st.markdown(
        """
    Selamat datang di aplikasi prediksi harga penutupan Bitcoin ₿🗠
    """
        """
Aplikasi ini merupakan sistem prediksi berbasis deep learning yang dirancang untuk membantu pengguna dalam memantau dan memperkirakan harga penutupan Bitcoin (BTC) selama 7 hari ke depan secara otomatis. 

Dengan memanfaatkan model LSTM (Long Short-Term Memory) dan integrasi API harga real-time, aplikasi ini menyajikan proyeksi harga , sehingga diharapkan pengguna mendapatkan informasi terhadap arah pasar.

---

Di dalam aplikasi ini, pengguna dapat mengakses 3 halaman utama, yaitu:
- **Beranda**: Gambaran umum aplikasi.
- **Tentang**: Menjelaskan tentang aplikasi, dataset dan model.
- **Prediksi Harga**: Menampilkan proyeksi harga Bitcoin 7 hari ke depan

---

Aplikasi ini bertujuan memberikan gambaran umum mengenai kemungkinan arah pergerakan harga Bitcoin berdasarkan data historis yang sudah dilatih. Namun demikian, prediksi yang dihasilkan bersifat estimatif dan tidak dimaksudkan sebagai acuan utama dalam pengambilan keputusan.
"""
    )

    st.warning(
        "⚠️ **Disclaimer:**  \n"
        "Prediksi harga yang ditampilkan hanya untuk tujuan informatif. Aplikasi ini tidak memberikan saran investasi. "
        "Selalu lakukan riset mandiri (DYOR) dan pertimbangkan risiko secara bijak sebelum mengambil keputusan finansial."
    )

# ===========================
#      TENTANG
# ===========================
elif page == "Tentang":
    st.title("Tentang")
    st.markdown(
        """
Aplikasi ini dibangun untuk memberikan gambaran mengenai potensi arah pergerakan harga penutupan Bitcoin (BTC) selama 7 hari ke depan secara otomatis. Dengan menggabungkan model deep learning yang dilatih dari data historis dan integrasi API CoinGecko dan Twelve Data untuk memperoleh harga terkini, aplikasi ini menyajikan prediksi penutupan harga Bitcoin yang bersifat estimatif.

---
### 📂 Penjelasan Data

Dataset utama berasal dari **Yahoo Finance**, mencakup data harian Bitcoin mulai dari **1 Januari 2015 hingga 11 Juli 2025**. Data disimpan dalam format `.csv`, diperoleh melalui skrip Python, dan dibersihkan secara manual sebelum digunakan.

Dataset terdiri dari **3.845 baris** dan mencakup kolom: `Date`, `Open`, `High`, `Low`, `Close`, dan `Volume`. Namun, hanya kolom **`Close` menjadi target utama prediksi, sementara `Volume`** dan fitur lainnya digunakan dalam eksplorasi serta pembentukan fitur prediktif. Untuk keperluan evaluasi, data historis dibagi dengan rasio 80% untuk pelatihan (training) dan 20% untuk pengujian (testing). 

Sementara itu, data real-time dari API CoinGecko sebagai sumber utama dan Twelve Data sebagai sumber cadangan untuk dimanfaatkan sebagai input aktual saat model melakukan prediksi harga terkini. Penggunaan API ini memungkinkan sistem menghasilkan prediksi harga penutupan Bitcoin secara dinamis tanpa perlu melakukan pelatihan ulang model setiap saat. Proses pelatihan model hanya dilakukan secara berkala, misalnya satu kali dalam sebulan, untuk memperbarui parameter model berdasarkan data terbaru yang tersedia, sehingga menjaga efisiensi komputasi dan tetap relevan dengan kondisi pasar.

---
### 🧠 Penjelasan Model

Model yang digunakan adalah **LSTM (Long Short-Term Memory)**, jenis jaringan saraf yang dirancang untuk memproses data urutan seperti harga waktu ke waktu. Model dilatih untuk memproyeksikan **log-return 7 hari ke depan** berdasarkan pola harga 14 hari terakhir.

Sebelum pelatihan, data diperkaya melalui proses *feature engineering* seperti indikator teknikal (EMA, RSI), return harian, dan fitur statistik lainnya. Seluruh fitur diskalakan menggunakan **MinMaxScaler** agar model lebih stabil saat belajar.

Model ini diuji serta dievaluasi menggunakan metrik seperti **MAE**, **RMSE**, dan **MAPE**. Prediksi akhir dikonversi dari log-return menjadi estimasi harga penutupan Bitcoin
"""
    )
    st.warning(
        "⚠️ **Disclaimer:**  \n"
        "Prediksi harga yang ditampilkan hanya untuk tujuan informatif. Aplikasi ini tidak memberikan saran investasi. "
        "Selalu lakukan riset mandiri (DYOR) dan pertimbangkan risiko secara bijak sebelum mengambil keputusan finansial."
    )

# ===========================
#     PREDIKSI HARGA
# ===========================
elif page == "Prediksi Harga":
    st.title("📊 Prediksi Pergerakan Penutupan Harga Bitcoin (BTC)")

    st.warning(
        "\u26a0\ufe0f Disclaimer: Prediksi harga Bitcoin yang ditampilkan di bawah ini dihasilkan oleh model kecerdasan buatan (AI) berbasis deep learning. "
        "Tidak ada jaminan akurasi sepenuhnya, dan harga dapat berubah secara dinamis karena banyak faktor eksternal. "
        "Selalu lakukan riset mandiri (DYOR), konsultasi dengan perencana keuangan atau pakar investasi jika diperlukan, "
        "dan ingat bahwa setiap keputusan investasi sepenuhnya menjadi tanggung jawab pribadi."
    )

    # # Inisialisasi refresh counter
    # if "refresh_count" not in st.session_state:
    #     st.session_state.refresh_count = 0

    # Ambil API key
    api_key = get_api_key("TWELVE_API_KEY")
    harga_btc, waktu_update, sumber_api = get_btc_api(api_key)

    # # Tombol refresh
    # if st.button("\U0001f503 Refresh Harga & Prediksi"):
    #     harga_btc, waktu_update, sumber_api = get_btc_api(api_key)
    #     st.session_state.refresh_count += 1
    # else:
    #     harga_btc, waktu_update, sumber_api = get_btc_api(api_key)

    if harga_btc is None or waktu_update is None:
        st.error(
            "\u274c Gagal mengambil harga dari API. Silakan cek koneksi internet atau coba beberapa saat lagi."
        )
        st.stop()

    st.metric(
        label="💰 Harga BTC Saat Ini (USD)",
        value=f"${harga_btc:,.2f}",
        # delta=waktu_update.strftime("%d %B %Y %H:%M:%S WIB (Terakhir Update)"),
        delta=f"{format_tanggal_indonesia(waktu_update)} {waktu_update.strftime('%H:%M:%S')} WIB (Terakhir Update)",
    )

    #  Info API dan sistem refresh
    st.caption(f"📡 API: **{sumber_api}**")
    st.caption(
        "⏳ Halaman akan diperbarui otomatis setiap 5 menit untuk menampilkan harga terbaru."
    )

    # Grafik 1: Harga Aktual (Real-Time 30 Hari)
    st.subheader("📈 Grafik Harga BTC Aktual 30 Hari Terakhir (Real-Time)")
    df_real = get_histori_btc(days=30)

    if not df_real.empty:
        fig_actual, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(
            df_real.index, df_real["price"], marker="o", linestyle="-", color="blue"
        )
        ax1.set_title("Harga Aktual Bitcoin (USD)")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga (USD)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=30)
        st.pyplot(fig_actual)
    else:
        st.error("❌ Gagal memuat data grafik harga BTC.")

    # Prediksi log-return 7 hari ke depan
    y_pred = model_predict(model, scaler, df_fe, harga_btc)
    hasil_prediksi = predict_harga_dari_logret(harga_btc, y_pred)

    # Tampilkan hasil prediksi
    st.subheader(" Tabel Prediksi Harga BTC 7 Hari ke Depan")
    tanggal_prediksi = [
        waktu_update + datetime.timedelta(days=i + 1)
        for i in range(len(hasil_prediksi))
    ]

    tabel_prediksi = pd.DataFrame(
        {
            "Tanggal (WIB)": [format_tanggal_indonesia(t) for t in tanggal_prediksi],
            "Harga Prediksi (USD)": [f"{h:,.2f}" for h in hasil_prediksi],
        }
    )

    st.table(tabel_prediksi.set_index("Tanggal (WIB)"))
    st.caption(
        "🔎Hasil prediksi bersifat estimasi dan bukan merupakan saran investasi. "
        "Selalu lakukan riset mandiri (DYOR)."
    )

    # Grafik 2: Prediksi Harga BTC
    st.subheader(" Grafik Prediksi Harga BTC 7 Hari ke Depan")
    fig_pred, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        tanggal_prediksi, hasil_prediksi, marker="o", linestyle="--", color="orange"
    )
    ax2.set_title("Prediksi Harga Bitcoin (USD) 7 Hari Mendatang")
    ax2.set_xlabel("Tanggal")
    ax2.set_ylabel("Harga (USD)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30)
    st.pyplot(fig_pred)
