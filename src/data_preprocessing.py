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
    df['Global_active_power_scaled'] = scaler.fit_transform(
        df[['Global_active_power']]
    )
    
    X, y = create_sequence(df['Global_active_power_scaled'].values, seq_length)
    
    X = X.reshape((-1, seq_length, 1))

    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler