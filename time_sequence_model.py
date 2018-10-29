# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:53:02 2018

@author: Ron
"""

from keras import layers
from keras import models
import numpy as np
from plotting import LivePlot
import preprocessing

def generate_signal(signal_length=1e6, noise=0.2, t=0, inject_anom=True):
    for t in np.arange(0, signal_length, 0.01):
        y = np.sin(2*np.pi*t) + np.random.uniform(-1,1) * noise
        # Inject anoamalies
        if inject_anom:
            if (np.random.binomial(1, 0.01))>0.5:
                y += np.random.uniform(-0.75,0.75)
        yield t, y

''' Createe training data '''
signal_generator = generate_signal(inject_anom=False)
''' Generate training data with ength of 1000'''
data = []
for i in range(1000):
    t, sig = next(signal_generator)
    data.append(sig)



train, test = preprocessing.split_data(data)

timesteps = 5
X_train, Y_train = preprocessing.arange_data_for_sequence_model(data, timesteps, lookforward=1)


def define_model(n_features, timesteps, batch_size=None, stateful=False, forward_pred=1):
    """
Defines and builds a model. This function can build both stateful and none stateful model. 
    -------------------------------------------------------------------------------------------
    args:
        n_features (int) - Number of features in the data
        timesteps (int) - number of look back timesteps the model uses to generate predictions
        batch_size (int) - model training batch size
        stateful (bool) - 
        forward_pred (int) - number of forward steps to forecast
    return:
        model - Keras SLTM model
    """
    
    inp = layers.Input(batch_shape=(batch_size, timesteps, n_features), name='imput')
    lstm = layers.LSTM(32, stateful=stateful, return_sequences=True, name='lstm0')(inp)
    lstm = layers.LSTM(forward_pred, stateful=stateful, return_sequences=False, name='lstm1')(lstm)
    dense = layers.Dense(n_features, name='out')(lstm)
    out = layers.RepeatVector(forward_pred)(dense)
    
    model = models.Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model



def get_prediction_model(weights_path, n_features, timesteps, statful=False):
    """
    Builds model that can accepts a batch size of 1 for predictions generations. The model 
    define a new model and loads the weights of the trained model
    -------------------------------------------------------------------------------------------
    args:
        n_features (int) - Number of features in the data
        timesteps (int) - number of look back timesteps the model uses to generate predictions
        batch_size (int) - model training batch size
        stateful (bool) - 
        forward_pred (int) - number of forward steps to forecast
    return:
        model - Keras SLTM model
    """
    
    model = define_model(n_features, timesteps, batch_size=1, stateful=stateful)
    model.load_weights(weights_path)
    return model

def build_train_model(X, Y, timesteps, batch_size=None, epochs=500, stateful=False, forward_pred=1):
    """
    A wrapper function that builds and train the model
    -------------------------------------------------------------------------------------------
    args:
        X (numpy array) - features collection for the model to train on
        Y (numpy array) - training targets (next steps in the time series data)
        timesteps (int) - number of times steps the model looks back to generate predictions
        batch_size (int) - training batch size
        epochs (int) - number of epochs to train on
        stateful (bool) - 
        forward_pred (int) - number of steps in to the future to forecast
        model
    return:
        prediction of the next value
    """    
    n_features = X.shape[-1]
    if stateful:
        model = define_model(n_features, timesteps, 
                             batch_size=batch_size, 
                             stateful=stateful, 
                             forward_pred=forward_pred)
        for e in range(epochs):
            print('Ephoc {}/{}'.format(e+1, epochs))
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, shuffle=False)
            model.reset_states()
    else:
        model = define_model(n_features, timesteps, 
                             batch_size=None, 
                             stateful=stateful, 
                             forward_pred=forward_pred)
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=False)
        
    model.save_weights('model_weights.h5') 
    return model, 'model_weights.h5', history


def predict(data_prev, model):
    """
        Gets previous data and model and generates next steps predictions
    -------------------------------------------------------------------------------------------
    args:
        data_rev - (list) - a list of timesteps of the previous data
        model
    return:
        prediction of the next value
    """
    timesteps = len(data_prev)
    n_features = data_prev[0].size
    pred = p_model.predict(np.array(data_prev).reshape(1,timesteps,n_features)).reshape(-1)
    return pred

def predictions_generator(data, model, timesteps):

    data_prev = np.array([next(data) for _ in range(timesteps)])
    t = data_prev[-1,0]
    X = data_prev[:,1]
    while True:
        pred = predict(X, model)
        yield t, X[-1], pred
        t, s = next(data)
        X.append(s)
        X.pop(0)
        
def amoanomalies_detector(data, model, timesteps):
    
    X = [np.array(next(data)[1:]) for _ in range(timesteps)]
    while True:
        pred = predict(X, model)
        t, s = next(data)
        e = np.power(np.subtract(pred, s), 2)
        yield t, s, pred, e
        X.append(s)
        X.pop(0)
        
    

n_features = X_train.shape[-1]
batch_size=128
epochs=500
stateful = False
forward_pred = Y_train.shape[1]
# Restart the signal generator with anomalies
signal_generator = generate_signal()


model, weights_path, history = build_train_model(X_train, Y_train, 
                                                 timesteps, batch_size, 
                                                 epochs, stateful=stateful,
                                                 forward_pred=forward_pred)
if stateful:
    p_model = get_prediction_model(weights_path, n_features, timesteps, batch_size=1, stateful=stateful)
else:
    p_model = model
    
#predict(, p_model) 

adg = amoanomalies_detector(signal_generator, p_model, 5)
ploter = LivePlot(adg)
ani = ploter.run()

