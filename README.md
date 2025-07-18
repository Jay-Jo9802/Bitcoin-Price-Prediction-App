# 📈 Prediksi Harga Penutupan Bitcoin 7 Hari ke Depan

> ⚠️ **Disclaimer:**
> Hasil prediksi yang ditampilkan di aplikasi ini bersifat estimatif dan dibuat berdasarkan model statistik serta data historis. Tidak ada jaminan akurasi atas hasil prediksi. Harga aset kripto seperti Bitcoin sangat fluktuatif dan berisiko. Keputusan investasi sepenuhnya merupakan tanggung jawab pengguna. Selalu lakukan riset mandiri (DYOR) dan konsultasi dengan ahli keuangan sebelum mengambil keputusan investasi.

---

## 🧠 Tentang Proyek

Aplikasi ini dirancang untuk memberikan estimasi arah pergerakan harga penutupan Bitcoin (BTC) selama tujuh hari ke depan secara otomatis. Model utama menggunakan pendekatan _Deep Learning_ berbasis **LSTM (Long Short-Term Memory)**, dilatih menggunakan data historis harga Bitcoin dari 2015 hingga 2025. Untuk mendukung prediksi terkini, model ini diintegrasikan dengan data real-time dari **API CoinGecko**, sehingga pengguna dapat melihat estimasi harga berbasis kondisi pasar saat ini.

---

## 📂 Dataset

- **Sumber Data Historis:** Yahoo Finance
- **Rentang Waktu:** 1 Januari 2015 – 11 Juli 2025
- **Jumlah Data:** 3.847 baris
- **Fitur yang Digunakan:**
  - `Close` (harga penutupan)
  - `Volume` (volume perdagangan)
- **Proses Pembagian Data:**
  - 80% untuk pelatihan (_training_)
  - 20% untuk pengujian (_testing_)

---

## 🔧 Preprocessing & Feature Engineering

Sebelum digunakan untuk pelatihan model, data historis diproses melalui tahap berikut:

- **Pembersihan Data:** Menghapus nilai null, anomali, dan konversi format tanggal
- **Feature Engineering:**
  - Indikator teknikal: EMA, RSI
  - Return harian dan log-return
  - Lag features (t-1 s.d t-n)
  - Rolling mean & std dev
- **Normalisasi:** Menggunakan `MinMaxScaler` dari Scikit-learn

---

## 🧮 Arsitektur Model LSTM

Model LSTM dirancang untuk memproyeksikan **log-return 7 hari ke depan** berdasarkan **input pola harga 14 hari sebelumnya**. Model terdiri dari:

- Layer LSTM (1 atau 2 layer)
- Dense output layer
- Aktivasi: ReLU / linear
- Loss: MAE
- Optimizer: Adam

Model telah dilatih dan dievaluasi menggunakan metrik:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

---

## 🌐 Integrasi API CoinGecko

Setelah model dilatih, prediksi real-time dilakukan dengan memanfaatkan **harga penutupan Bitcoin terkini** dari **CoinGecko API**. Data ini digunakan sebagai input aktual (_current close_) untuk memproyeksikan estimasi harga 7 hari ke depan.

Fallback API seperti Twelve Data dapat disiapkan jika diperlukan.

> Model tidak dilatih ulang setiap saat. Proses training dilakukan secara berkala (misal 1x sebulan) untuk memperbarui parameter model berdasarkan data terbaru.

---

## 💻 Struktur Halaman Aplikasi

Aplikasi dibangun menggunakan **Streamlit** dan terdiri dari 4 halaman utama:

1. **Beranda:** Gambaran umum aplikasi dan navigasi antar halaman
2. **Tentang:** Penjelasan lengkap tentang data, model, dan proses prediksi
3. **EDA (Exploratory Data Analysis):** Visualisasi dan statistik historis harga Bitcoin
4. **Prediksi Harga:** Tampilan harga terkini dari API CoinGecko dan prediksi 7 hari ke depan

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/Jay-Jo9802/Bitcoin-Price-Prediction-App.git
