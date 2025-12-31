from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def get_train_test_val(X, y, subject_ids, test_size=0.2, val_size=0.2, random_state=42, normalize=True):
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    print(f"y.shape: {y.shape}")
    print(f"X.shape: {X.shape}")

    unique_subjects = np.unique(subject_ids)

    subj_train_val, subj_test = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )

    subj_train, subj_val = train_test_split(
        subj_train_val, test_size=val_size / (1 - test_size), random_state=random_state
    )

    mask_train = np.isin(subject_ids, subj_train)
    mask_val = np.isin(subject_ids, subj_val)
    mask_test = np.isin(subject_ids, subj_test)

    X_train, y_train, sid_train = X[mask_train], y[mask_train], subject_ids[mask_train]
    X_val, y_val, sid_val = X[mask_val], y[mask_val], subject_ids[mask_val]
    X_test, y_test, sid_test = X[mask_test], y[mask_test], subject_ids[mask_test]

    return X_train, y_train, X_val, y_val, X_test, y_test, sid_train, sid_val, sid_test

def apply_rowwise_minmax_scaling(df):
    features = df.columns.drop(['Unnamed', 'y', 'second_ids', 'subject_ids'])
    X_scaled = []
    for _, row in df[features].iterrows():
        scaler = MinMaxScaler()
        scaled_row = scaler.fit_transform(row.values.reshape(-1, 1)).flatten()
        X_scaled.append(scaled_row)
    df.loc[:, features] = X_scaled
    return df

def get_features_and_labels(data, scale=False):
    y = data['y']
    X = data[[x for x in data.columns if x.startswith('X')]]
    if scale == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return y, X

