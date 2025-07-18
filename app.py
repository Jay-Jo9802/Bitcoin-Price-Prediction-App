#
import streamlit as st
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
    eda_summary_harian,
    eda_summary_bulanan,
    eda_summary_tahunan,
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

# Setup halaman
st.set_page_config(page_title="Prediksi Harga BTC", page_icon="💰")

# Navigasi
st.sidebar.markdown(
    "<h2 style='text-align: left;'>Navigasi</h2>", unsafe_allow_html=True
)
page = st.sidebar.selectbox(
    "Pilih Halaman", ["Beranda", "Tentang", "EDA", "Prediksi Harga"]
)


# Ambil harga BTC dari API
@st.cache_data(ttl=300)
def get_btc_api(api_key: str):
    try:
        # url = "https://api.coingecko.com/api/v3/simple/price"
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

Dengan memanfaatkan model LSTM (Long Short-Term Memory) dan data historis yang diperoleh dari sumber terpercaya seperti Yahoo Finance dan CoinGecko, aplikasi ini tidak hanya memberikan prediksi harga, tetapi juga menyediakan analisis statistik dan visualisasi untuk mendukung pemahaman terhadap tren pasar.

Di dalam aplikasi ini, pengguna dapat mengakses empat halaman utama, yaitu:

- **Beranda**: Menyajikan gambaran umum tentang tujuan aplikasi dan fitur-fitur utamanya.
- **Tentang**: Menjelaskan metodologi yang digunakan, mulai dari proses preprocessing, teknik feature engineering, hingga arsitektur model LSTM yang diterapkan.
- **EDA (Exploratory Data Analysis)**: Memberikan analisis statistik dan visualisasi harga Bitcoin berdasarkan data historis, mencakup pola harian, bulanan, dan tahunan.
- **Prediksi Harga**: Menampilkan harga BTC terkini dari API CoinGecko, dan memproyeksikan harga 7 hari ke depan berdasarkan output model LSTM.

Aplikasi ini bertujuan memberikan gambaran umum mengenai kemungkinan arah pergerakan harga Bitcoin berdasarkan data historis. Namun demikian, prediksi yang dihasilkan bersifat estimatif dan tidak dimaksudkan sebagai acuan utama dalam pengambilan keputusan.
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
Aplikasi ini dikembangkan untuk memberikan gambaran mengenai potensi arah pergerakan harga Bitcoin (BTC) selama tujuh hari ke depan secara otomatis. Dengan menggabungkan model deep learning dan data historis, aplikasi ini menyajikan prediksi harga yang bersifat estimatif dan dapat digunakan sebagai bahan pertimbangan tambahan.

---
### 📂 Penjelasan Data

Dataset utama berasal dari **Yahoo Finance**, mencakup data harian Bitcoin mulai dari **1 Januari 2015 hingga 11 Juli 2025**. Data disimpan dalam format `.csv`, diperoleh melalui skrip Python, dan dibersihkan secara manual sebelum digunakan.

Dataset terdiri dari **3.847 baris** dan mencakup kolom: `Date`, `Open`, `High`, `Low`, `Close`, dan `Volume`. Namun, hanya kolom **`Close` dan `Volume`** yang digunakan dalam pelatihan karena keduanya paling relevan dalam memprediksi harga.

Untuk keperluan evaluasi, data dibagi menggunakan rasio **80% training** dan **20% testing**. Sementara itu, **data real-time dari API CoinGecko** digunakan sebagai input aktual saat model melakukan prediksi terkini.

---
### 🧠 Penjelasan Model

Model yang digunakan adalah **LSTM (Long Short-Term Memory)**, jenis jaringan saraf yang dirancang untuk memproses data urutan seperti harga waktu ke waktu. Model dilatih untuk memproyeksikan **log-return 7 hari ke depan** berdasarkan pola harga 14 hari terakhir.

Sebelum pelatihan, data diperkaya melalui proses *feature engineering* seperti indikator teknikal (EMA, RSI), return harian, dan fitur statistik lainnya. Seluruh fitur diskalakan menggunakan **MinMaxScaler** agar model lebih stabil saat belajar.

Model ini dievaluasi menggunakan metrik seperti **MAE**, **RMSE**, **MAPE**, dan **R²**. Prediksi akhir dikonversi dari log-return menjadi estimasi harga penutupan berbasis harga terakhir yang diketahui.
"""
    )
    st.warning(
        "⚠️ **Disclaimer:**  \n"
        "Prediksi harga yang ditampilkan hanya untuk tujuan informatif. Aplikasi ini tidak memberikan saran investasi. "
        "Selalu lakukan riset mandiri (DYOR) dan pertimbangkan risiko secara bijak sebelum mengambil keputusan finansial."
    )

# ===========================
#           EDA
# ===========================
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA) BTC")
    # Overview Bitcoin (Candlestick + Volume) dengan opsi interval
    import plotly.graph_objects as go

    interval_map = {
        "1 Hari (1D)": "1",
        "7 Hari (7D)": "7",
        "1 Bulan (1M)": "30",
        "1 Tahun (1Y)": "365",
    }
    interval_label = st.selectbox(
        "Pilih Interval Candlestick", list(interval_map.keys()), index=3
    )
    interval_days = interval_map[interval_label]

    def get_ohlc_volume(days):
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {"vs_currency": "usd", "days": days}
        try:
            res = requests.get(url, params=params)
            ohlc = res.json()
            df_ohlc = pd.DataFrame.from_records(
                ohlc, columns=["timestamp", "open", "high", "low", "close"]
            )
            df_ohlc["date"] = pd.to_datetime(df_ohlc["timestamp"], unit="ms")
            df_ohlc.set_index("date", inplace=True)
            # Ambil volume
            url_vol = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params_vol = {"vs_currency": "usd", "days": days, "interval": "daily"}
            res_vol = requests.get(url_vol, params=params_vol)
            data_vol = res_vol.json()
            volumes = data_vol["total_volumes"]
            df_vol = pd.DataFrame.from_records(volumes, columns=["timestamp", "volume"])
            df_vol["date"] = pd.to_datetime(df_vol["timestamp"], unit="ms")
            df_vol.set_index("date", inplace=True)
            # Gabungkan volume ke df_ohlc (align index)
            df_ohlc["volume"] = df_vol["volume"].reindex(
                df_ohlc.index, method="nearest"
            )
            return df_ohlc
        except Exception as e:
            return None

    df_ohlc = get_ohlc_volume(interval_days)
    if df_ohlc is not None and not df_ohlc.empty:
        st.subheader(f"Overview Bitcoin (Candlestick & Volume) - {interval_label}")
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df_ohlc.index,
                open=df_ohlc["open"],
                high=df_ohlc["high"],
                low=df_ohlc["low"],
                close=df_ohlc["close"],
                name="OHLC",
            )
        )
        fig.add_trace(
            go.Bar(
                x=df_ohlc.index,
                y=df_ohlc["volume"],
                marker_color="lightblue",
                name="Volume",
                yaxis="y2",
                opacity=0.3,
            )
        )
        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Harga (USD)",
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h"),
            height=600,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Gagal mengambil data overview Bitcoin dari CoinGecko.")
    st.markdown(
        '<div style="font-size:10px; color:gray; text-align:right;">Sumber data: CoinGecko</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Rata-rata Harga Close per Hari")
    _, chart, insight = eda_summary_harian(df_fe)
    st.pyplot(chart)
    st.write(insight)
    st.subheader("Rata-rata per Bulan")
    _, chart, insight = eda_summary_bulanan(df_fe)
    st.pyplot(chart)
    st.write(insight)
    st.subheader("Rata-rata per Tahun")
    _, chart, insight = eda_summary_tahunan(df_fe)
    st.pyplot(chart)
    st.write(insight)
    st.subheader("Tren Harga Close")
    st.line_chart(df_fe["Close"])
    st.subheader("Volume Transaksi")
    st.line_chart(df_fe["Volume"])


# ===========================
#     PREDIKSI HARGA
# ===========================
elif page == "Prediksi Harga":
    st.title("📊 Prediksi Pergerakan Penutupan Harga Bitcoin (BTC)")

    st.warning(
        "⚠️ Disclaimer: Prediksi harga Bitcoin yang ditampilkan di bawah ini dihasilkan oleh model kecerdasan buatan (AI) berbasis deep learning. "
        "Tidak ada jaminan akurasi sepenuhnya, dan harga dapat berubah secara dinamis karena banyak faktor eksternal. "
        "Selalu lakukan riset mandiri (DYOR), konsultasi dengan perencana keuangan atau pakar investasi jika diperlukan, "
        "dan ingat bahwa setiap keputusan investasi sepenuhnya menjadi tanggung jawab pribadi."
    )

    # Ambil API key secara aman
    api_key = get_api_key("TWELVE_API_KEY")

    # Refresh harga
    if st.button("🔄 Refresh Harga & Prediksi"):
        harga_btc, waktu_update, sumber_api = get_btc_api(api_key)
    else:
        harga_btc, waktu_update, sumber_api = get_btc_api(api_key)

    if harga_btc is None or waktu_update is None:
        st.error(
            "Gagal mengambil harga dari API. Silakan cek koneksi internet atau coba beberapa saat lagi."
        )
        st.stop()

    # Tampilkan harga BTC sekarang
    st.metric(
        "💰 Harga BTC Saat Ini (USD)",
        f"${harga_btc:,.2f}",
        waktu_update.strftime("%d %B %Y %H:%M:%S WIB"),
    )
    st.caption(f"📡 API saat ini digunakan: **{sumber_api}**")

    # Prediksi log-return 7 hari ke depan
    y_pred = model_predict(model, scaler, df_fe, harga_btc)
    hasil_prediksi = predict_harga_dari_logret(harga_btc, y_pred)

    # Tampilkan tabel prediksi
    st.subheader("📈 Prediksi Harga 7 Hari ke Depan")
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
        "⚠️ Prediksi harga hanya bersifat estimasi, bukan saran investasi. Lakukan riset mandiri (DYOR)."
    )

    # Ambil data harga aktual terakhir 14 hari (atau sesuai kebutuhan)
    n_hist = 14
    hist_dates = df_fe.index[-n_hist:]
    hist_prices = df_fe["Close"][-n_hist:]

    # Gabungkan tanggal dan harga aktual dengan prediksi
    all_dates = list(hist_dates) + tanggal_prediksi
    all_prices = list(hist_prices) + list(hasil_prediksi)

    # Buat label untuk legend
    actual_label = "Harga Aktual"
    pred_label = "Prediksi Harga"

    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot harga aktual
    ax.plot(
        hist_dates,
        hist_prices,
        marker="o",
        linestyle="-",
        color="blue",
        label=actual_label,
    )
    # Plot harga prediksi (mulai dari harga aktual terakhir)
    ax.plot(
        [hist_dates[-1]] + tanggal_prediksi,
        [hist_prices[-1]] + list(hasil_prediksi),
        marker="o",
        linestyle="--",
        color="orange",
        label=pred_label,
    )

    # Garis vertikal pemisah
    x_vline = hist_dates[-1]
    # Jika x_vline masih DatetimeIndex, ambil elemen terakhir dan konversi ke datetime
    if isinstance(x_vline, (pd.DatetimeIndex, pd.Index)):
        x_vline = x_vline[-1]
    if isinstance(x_vline, pd.Timestamp):
        x_vline = x_vline.to_pydatetime()
    elif not isinstance(x_vline, datetime.datetime):
        x_vline = pd.to_datetime(x_vline)
        if isinstance(x_vline, pd.Timestamp):
            x_vline = x_vline.to_pydatetime()
        elif not isinstance(x_vline, datetime.datetime):
            raise ValueError("x_vline harus berupa datetime.datetime")
    # Pastikan x_vline bukan DatetimeIndex lagi
    if isinstance(x_vline, (pd.DatetimeIndex, pd.Index)):
        raise ValueError("x_vline tidak boleh DatetimeIndex")
    # Konversi ke float untuk axvline
    x_vline_float = mdates.date2num(x_vline)
    if hasattr(x_vline_float, "__len__") and not isinstance(x_vline_float, str):
        x_vline_float = x_vline_float[-1]
    x_vline_float = float(x_vline_float)
    ax.axvline(
        x=x_vline_float, color="gray", linestyle=":", alpha=0.7, label="Awal Prediksi"
    )

    ax.set_title("Prediksi Harga BTC 7 Hari ke Depan vs Harga Aktual")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (USD)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30)
    ax.legend()
    st.pyplot(fig)
