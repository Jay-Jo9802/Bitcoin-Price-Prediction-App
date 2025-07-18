import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ Konfigurasi ------------------
lookback = 14
features = [
    'Close', 'Volume', 'EMA_10', 'EMA_50', 'RSI_14',
    'log_return', 'return',
]
features += [f'Close_mean_{w}' for w in [5,10,14,20]]
features += [f'Close_std_{w}' for w in [5,10,14,20]]
features += [f'Volume_mean_{w}' for w in [5,10,14,20]]
features += [f'Close_lag_{i}' for i in range(1, 11)]

# ------------------ Loader ------------------
def load_scaler(path):
    return joblib.load(path)

def load_dataset(path):
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    return df

# ------------------ Feature Engineering ------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def feature_engineering(df):
    df = df.copy()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return'] = df['Close'].pct_change()
    for w in [5,10,14,20]:
        df[f'Close_mean_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'Close_std_{w}'] = df['Close'].rolling(window=w).std()
        df[f'Volume_mean_{w}'] = df['Volume'].rolling(window=w).mean()
    for i in range(1, 11):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    df = df.dropna()
    return df

# ------------------ Preprocess & Window Update ------------------
def make_window_custom(df_fe, scaler, harga_baru):
    df_temp = df_fe.copy()
    new_row = df_temp.iloc[[-1]].copy()
    new_row['Close'] = harga_baru

    new_row['EMA_10'] = pd.Series(pd.concat([df_temp['Close'], new_row['Close']]).ewm(span=10).mean()).iloc[-1]
    new_row['EMA_50'] = pd.Series(pd.concat([df_temp['Close'], new_row['Close']]).ewm(span=50).mean()).iloc[-1]

    close_temp = pd.concat([df_temp['Close'], new_row['Close']])
    delta = close_temp.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    new_row['RSI_14'] = 100 - (100 / (1 + pd.Series(rs).iloc[-1]))

    new_row['log_return'] = np.log(harga_baru / df_temp['Close'].iloc[-1])
    new_row['return'] = (harga_baru - df_temp['Close'].iloc[-1]) / df_temp['Close'].iloc[-1]

    for w in [5,10,14,20]:
        new_row[f'Close_mean_{w}'] = pd.Series(pd.concat([df_temp['Close'], new_row['Close']]).rolling(window=w).mean()).iloc[-1]
        new_row[f'Close_std_{w}'] = pd.Series(pd.concat([df_temp['Close'], new_row['Close']]).rolling(window=w).std()).iloc[-1]
        new_row[f'Volume_mean_{w}'] = pd.Series(pd.concat([df_temp['Volume'], new_row['Volume']]).rolling(window=w).mean()).iloc[-1]

    for i in range(1, 11):
        if i == 1:
            new_row[f'Close_lag_{i}'] = df_temp['Close'].iloc[-1]
        else:
            new_row[f'Close_lag_{i}'] = df_temp[f'Close_lag_{i-1}'].iloc[-1]

    df_temp = pd.concat([df_temp, new_row])
    X_input = df_temp[features].iloc[-lookback:].values
    X_input = scaler.transform(X_input)
    return X_input.reshape(1, lookback, len(features)), df_temp

# ------------------ Prediksi Multi-step ------------------
def model_predict(model, scaler, df_fe, harga_awal, horizon=7):
    harga_pred_logret = []
    harga_sekarang = harga_awal
    df_temp = df_fe.copy()

    for _ in range(horizon):
        X_input, df_temp = make_window_custom(df_temp, scaler, harga_sekarang)
        y_logret = model.predict(X_input, verbose=0)[0][0]  # ambil 1 log-return
        harga_pred_logret.append(y_logret)
        harga_sekarang = harga_sekarang * np.exp(y_logret)

    return harga_pred_logret

# ------------------ Konversi Log-Return ke Harga ------------------
def predict_harga_dari_logret(harga_awal, logret_pred):
    hasil = [harga_awal]
    for logret in logret_pred:
        hasil.append(hasil[-1] * np.exp(logret))
    return hasil[1:]

# ------------------ EDA ------------------
def eda_summary_harian(df):
    df = df.copy()
    df['day_name'] = df.index.day_name(locale='English')
    hari_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    eda = df.groupby('day_name')['Close'].mean().reindex(hari_order)
    fig, ax = plt.subplots()
    eda.plot(kind='bar', ax=ax)
    ax.set_ylabel('Rata-rata Harga Close')
    ax.set_title('Rata-rata Harga Close per Hari')
    insight = f"Hari tertinggi: {eda.idxmax()} ({eda.max():,.2f}), terendah: {eda.idxmin()} ({eda.min():,.2f})"
    return eda, fig, insight

def eda_summary_bulanan(df):
    df = df.copy()
    df['month_name'] = df.index.month_name(locale='English')
    bulan_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    eda = df.groupby('month_name')['Close'].mean().reindex(bulan_order)
    fig, ax = plt.subplots()
    eda.plot(kind='bar', ax=ax)
    ax.set_ylabel('Rata-rata Harga Close')
    ax.set_title('Rata-rata Harga Close per Bulan')
    insight = f"Bulan tertinggi: {eda.idxmax()} ({eda.max():,.2f}), terendah: {eda.idxmin()} ({eda.min():,.2f})"
    return eda, fig, insight

def eda_summary_tahunan(df):
    eda = df.groupby(df.index.year)['Close'].mean()
    fig, ax = plt.subplots()
    eda.plot(kind='bar', ax=ax)
    ax.set_ylabel('Rata-rata Harga Close')
    ax.set_title('Rata-rata Harga Close per Tahun')
    insight = f"Tahun tertinggi: {eda.idxmax()} ({eda.max():,.2f}), terendah: {eda.idxmin()} ({eda.min():,.2f})"
    return eda, fig, insight
