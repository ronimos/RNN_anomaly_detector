# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:45:42 2018

@author: Ron Simenhois
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import os
import warnings

warnings.simplefilter('ignore', category=FutureWarning)

MISSING_VALUE_MARKER = -5

def fill_mark_missing_timesteps(df):
    """
    Get a dataframe with index of times. Check for missing data and return
    a dataframe with number of rows expand to the number of time steps
    within the time frame between the min and the max time and the missing 
    data filled with the column's mean values and a map numpy of the 
    missing values
    -----------------------------------------------------------------------
    args:
        df - (Pandas dataframe) - The index of the dataframe needs to me the times
                                  of the timesteps
    returns:
        filled (Pandas dataframe) - An expand version of the original dataframe
                                    with a row for every timestep and the missing 
                                    data filled with the column's mean
        missing (Numpy arrray, dtype bool) - A Boolean map with the same dimensions
                                             as the filled dataframe. The missing
                                             map maps the locations of the missing / 
                                             inserted values
    """    
    filled = df.copy()
    filled.index = pd.to_datetime(filled.index)
    filled.sort_index()
    times = pd.Series(pd.to_datetime(filled.index))       
    delta = (times - times.shift(1)).min()    
    filled = filled.resample(delta).asfreq()
    missing = np.zeros_like(filled.values, dtype=bool)
    missing[filled.isnull().values]=True

    return filled, missing


def scale(df, missing, mark_missing=True):
    """
    Scales the data with a Min Max scaler while keeping missing data marked
    -----------------------------------------------------------------------
    args:
        df (Pandas dataframe) - The index of the dataframe needs to me the times
                                  of the timesteps
        missing (numpy array) - a map of the missing data in the original dataset
        mark_missing (boolean) - a flag to mark if missing data should be set to the mean
                                 or marked with values outside of the database range (-3)
    returns:
        scaled (Numpy arrray) - The data after Min Max scaling to [0,1]
    """        
    if np.any(df.isnull()):
        df, missing = fill_mark_missing_timesteps(df)
        
    scaler_file_path = 'MinMax_scaler.save'
    if os.path.isfile(scaler_file_path):
        scaler = joblib.load(scaler_file_path)
    else:
        scaler = MinMaxScaler()
        scaler.fit_transform(df.values)
    scaled = scaler.transform(df.values)
    if mark_missing:
        scaled[missing] = MISSING_VALUE_MARKER
        
    return scaled
    

def split_data(data, test_size=0.25):
    data = pd.DataFrame(data)
    train_size = int((1-test_size)*len(data))
    train = data[:train_size]
    test = data[train_size:]
    
    return train, test

def arange_data_for_sequence_model(data, lookback=2, lookforward=1):
    """
    Re arange the data two into 3D array the size of :
    (number of items X timesteps X number of features). One numpay array of the
    past timesteps data (X) and the other is the predictions traget 
    -----------------------------------------------------------------------
    args:
        data (list) - list of numpy arrays 
        lookback (int) - number of timesteps to look back 
        lookforward (int) - number of timesteps in the future to predict 
    returns:
        X (Numpy arrray) - the past data in 3D array
        Y (Numpy arrray) - training tragets in 3D array
    """            
    data = pd.DataFrame(data)
    n_features = data.shape[1]
    agg = pd.concat([data.shift(i) for i in range(lookback, -lookforward, -1)], axis=1)
    agg.dropna(inplace=True)
    X = agg.iloc[:,:-lookforward*n_features].values
    Y = agg.iloc[:,-lookforward*n_features:].values
    X = X.reshape(-1, lookback, n_features)
    Y = Y.reshape(-1, lookforward ,n_features)
        
    return X, Y
