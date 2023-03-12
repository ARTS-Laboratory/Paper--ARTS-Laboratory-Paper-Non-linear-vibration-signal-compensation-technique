import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
"""
Trying to get timestamps for each forward step
"""
#%% load dataset
"""
Test datasets:
    0-1 Hz, 0-5 Hz, 0-10 Hz, 0-20 Hz
Train datasets:
    0-1 Hz, 1-2 Hz, 2-3 Hz, 3-4 Hz, 4-5 Hz, 5-6 Hz, 6-7 Hz, 7-8 Hz, 8-9 Hz
    9-10 Hz, 10-11 Hz, 11-12 Hz, 12-13 Hz, 13-14 Hz, 14-15 Hz, 15-16 Hz,
    16-17 Hz, 17-18 Hz, 18-19 Hz, 19-20 Hz, 20-21 Hz, 0-20 Hz
"""
X_test = np.load("./dataset/X_test.npy")[-1]
X_test = X_test[:X_test.size//2].reshape(1, -1)

X_test = np.expand_dims(X_test, -1)
X_test_step = X_test.reshape(-1, 1, 1)
#%%
class Timestamps(keras.callbacks.Callback):
    
    def __init__(self, n_timesteps=100):
        self.end_timestamps = np.zeros(n_timesteps)
        self.begin_timestamps = np.zeros(n_timesteps)
        self.i = 0
    
    def on_predict_batch_begin(self, batch, logs=None):
        self.begin_timestamps[self.i] = tf.timestamp();
    
    def on_predict_batch_end(self, batch, logs=None):
        self.end_timestamps[self.i] = tf.timestamp(); self.i+=1
    
    def get_timestamps(self):
        return (self.begin_timestamps, self.end_timestamps)
    
    def get_times(self):
        return self.end_timestamps - self.begin_timestamps
#%%
model = keras.models.load_model("./model_saves/model")
#%%
UNITS = 50

new_model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=True, 
                      batch_input_shape=(1, None, 1), 
                      trainable=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, use_bias=True, trainable=True))
])

for l1, l2 in zip(model.layers, new_model.layers):
    l2.set_weights(l1.get_weights())
model = new_model
timestamps = Timestamps(n_timesteps=X_test.shape[1])

model = new_model
#%% run predictions
model.predict(X_test_step,
              batch_size=1, 
              callbacks=[timestamps],
)
#%% make a distribution
dt = timestamps.get_times()

plt.plot(dt[1:])