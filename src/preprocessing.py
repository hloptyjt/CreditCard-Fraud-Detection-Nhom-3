import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(path):
    """Đọc dữ liệu từ file CSV"""
    df = pd.read_csv(path)
    print(f"Kích thước dữ liệu: {df.shape}")
    return df

def scale_time_amount(X):
    """Scale riêng cột Time và Amount"""
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    return X, scaler

def scale_full(X):
    """Scale toàn bộ dữ liệu"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Chia train/test với stratify"""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)