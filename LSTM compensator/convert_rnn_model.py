#%%
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
"""
Load model, convert to RNN model (so that it can be used with TFLite)
"""
#%% load model
model = keras.models.load_model("./model_saves/model")
#%% create RNN model
UNITS = 50
rnn_model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      input_shape=(None, 1),
                      trainable=True),
    keras.layers.SimpleRNN(1, 
                      activation=None,
                      use_bias=True,
                      return_sequences=True)
])

rnn_model.layers[0].set_weights(model.layers[0].get_weights())

dense_weights, dense_bias = model.layers[1].get_weights()
rnn_weights = (dense_weights, np.zeros((1,1)), dense_bias)

rnn_model.layers[1].set_weights(rnn_weights)
#%% load dataset
X_test_full = np.load("./dataset/X_test.npy")
Y_test_full = np.load("./dataset/Y_test.npy")

X_test = X_test_full[-1]
Y_test = Y_test_full[-1]

X_test = X_test[:X_test.size//2].reshape(1, -1)
Y_test = Y_test[:Y_test.size//2].reshape(1, -1)
#%% 
model_p = model.predict(X_test)
rnn_model_p = rnn_model.predict(X_test)

dif = np.max(np.abs(model_p - rnn_model_p))
print("max difference: %f"%dif)
#%% save RNN model
rnn_model.save("./model_saves/rnn_model")