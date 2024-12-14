import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os


def load_real_data(data_name):
    if data_name == 'ihdp':
        X, Y0, Y1 = ihdp()
    elif data_name == 'boston':
        X, y = pickle.load(open(Path(__file__).parents[1] / "data/boston_dataset.pkl", "rb"))
        Y1 = Y0 = y
    elif data_name == 'lalonde':
        X, y = lalonde()
        Y1 = Y0 = y
    elif data_name == 'weather':
        X, Y0, Y1 = weather()
    elif 'acic' in data_name:
        X, Y0, Y1 = acic()
    return X, Y0, Y1


def train_test_split(X, Y0, Y1, test_ratio, seed):
    npoints, _ = X.shape
    df = pd.DataFrame({'X': X.tolist(), 'Y0': Y0, 'Y1': Y1})
    df = df.sample(frac=1.0, replace=False, ignore_index=True, random_state=seed)
    train_data = df.iloc[int(npoints * test_ratio):]
    test_data = df.iloc[:int(npoints * test_ratio)]
    return train_data, test_data


def ihdp():
    # column names: ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ] + feautres
    ihdp_dataset = pd.read_csv(Path(__file__).parents[1] / 'data/ihdp_npci_1.csv', header=None)
    data_numpy = ihdp_dataset.to_numpy()
    X = data_numpy[:, 5:]
    Y1 = np.zeros(len(X))
    Y0 = np.zeros(len(X))
    for i in range(0, len(X)):
        if data_numpy[i, 0] == 1:
            Y0[i] = data_numpy[i, 2]
            Y1[i] = data_numpy[i, 1]
        else:
            Y0[i] = data_numpy[i, 1]
            Y1[i] = data_numpy[i, 2]
    return X, Y0, Y1


def lalonde():
    data = pd.read_csv(Path(__file__).parents[1] / 'data/lalonde.csv')
    labelled_data = data.to_numpy()
    _, d = labelled_data.shape
    cols = [True for i in range(0, d)]
    cols[8], cols[11] = False, False
    X = labelled_data[:, np.array(cols)]
    cols = [False for i in range(0, d)]
    cols[8] = True
    y = labelled_data[:, np.array(cols)]
    y = np.array(y[:, 0])
    return X, y

def acic():
    data = pd.read_csv(Path(__file__).parents[1] / 'data' / 'acic1.csv')
    label = pd.read_csv(Path(__file__).parents[1] / 'data' / 'acic1_cf.csv')
    data = data.drop(columns=['Y', 'A'])
    X = np.array(data)
    Y0 = np.array(label['EY0_i'])
    Y1 = np.array(label['EY1_i'])
    return X, Y0, Y1

def weather():
    data = pd.read_csv(Path(__file__).parents[1] / 'data' / 'weather.csv')
    X = data[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                 'Loud Cover', 'Pressure (millibars)']]
    X = np.array(X)
    Y0 = np.array(data['Visibility (km)'])
    Y1 = np.array(data['Visibility (km)'])
    return X, Y0, Y1
