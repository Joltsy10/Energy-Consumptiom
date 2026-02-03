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

def create_sequence(data, seq_length):

    xs,ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs),np.array(ys)

def prepare_data(df, seq_length=24):

    scaler = MinMaxScaler()
    
    
    X, y = create_sequence(df['Global_active_power'].values, seq_length)

    
    
    X = X.reshape((-1, seq_length, 1))

    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    scaler.fit(X_train.reshape(-1,1))

    X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), scaler