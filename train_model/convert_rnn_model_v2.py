#%%
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
"""
Load model, convert to RNN model (so that it can be used with TFLite)
 - use unroll=True
"""
#%% load model
model = keras.models.load_model("./model_saves/model")
#%% create RNN model
UNITS = 50
rnn_model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True,
                      stateful=True,
                      batch_input_shape=(1, 20, 1),
                      trainable=True,
                      unroll=True),
    keras.layers.SimpleRNN(1,
                      activation=None,
                      use_bias=True,
                      return_sequences=True,
                      unroll=True,)
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

X_test = X_test[:X_test.size//2].reshape(1, -1, 1)
Y_test = Y_test[:Y_test.size//2].reshape(1, -1, 1)

X_test_reshape = X_test.reshape(-1, 20, 1)
#%% 
model_p = model.predict(X_test)
rnn_model_p = rnn_model.predict(X_test_reshape, batch_size=1)

dif = np.max(np.abs(model_p.flatten() - rnn_model_p.flatten()))
print("max difference: %f"%dif)

plt.figure()
plt.plot(model_p.flatten())
plt.plot(rnn_model_p.flatten())
#%% save RNN model
rnn_model.save("./model_saves/rnn_model")