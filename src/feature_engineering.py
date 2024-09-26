import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_data(train_data, val_data, test_data, numeric_columns):
    
    scaler = MinMaxScaler().fit(train_data[numeric_columns])
    
    train_data[numeric_columns] = scaler.transform(train_data[numeric_columns])
    val_data[numeric_columns] = scaler.transform(val_data[numeric_columns])
    test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])
    
    return train_data, val_data, test_data

def extract_date_features(df):
    
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    return df