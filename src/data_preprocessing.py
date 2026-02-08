import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):

    df = pd.read_csv(filepath, 
                     sep = ';',
                     parse_dates={'datetime' : ['Date','Time']},
                     low_memory= False,
                     na_values=['nan' ,'?'])
    
    df.dropna(inplace=True)
    return df

def create_sequence(data, target, seq_length):

    xs,ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs),np.array(ys)

def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 6.0)
    df["day_cos"] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 6.0)
    df["month_sin"] = np.sin(2 * np.pi * df['datetime'].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df['datetime'].dt.month / 12.0)

    df['weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(float)

    return df

def prepare_data(df, seq_length=24):

    df = add_time_features(df)

    feature_cols = [
        'Global_active_power',
        'hour_sin',
        'hour_cos',
        'weekend',
        'day_sin',
        'day_cos',
        'month_sin',
        'month_cos'
    ]

    data = df[feature_cols].values
    target = df['Global_active_power'].values
    
    X, y = create_sequence(data, target, seq_length)

    num_features = X.shape[-1]

    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    X_train_reshaped = X_train.reshape(-1, num_features)
    X_scaler = MinMaxScaler()   
    X_scaler.fit(X_train_reshaped)

    X_train_scaled = X_scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_scaled = X_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), X_scaler, y_scaler