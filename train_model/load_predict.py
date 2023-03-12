import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
"""
Verifying the model was saved correctly and making predictions on training and
testing datasets. Also getting a distribution
"""
""" signal to noise ratio """
def signaltonoise(sig, noisy_signal, dB=True):
    noise = sig - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(sig)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
#%%
model = keras.models.load_model("./model_saves/model")

"""
Test datasets:
    0-10 Hz
Train datasets:
    0-1 Hz, 1-2 Hz, 2-3 Hz, 3-4 Hz, 4-5 Hz, 0-10 Hz
"""
X_train = np.load("./dataset/V4/X_train.npy").reshape(10, -1, 1)
Y_train = np.load("./dataset/V4/Y_train.npy").reshape(10, -1, 1)
X_test = np.load("./dataset/V5/X_test.npy").reshape(1, -1, 1)
Y_test = np.load("./dataset/V5/Y_test.npy").reshape(1, -1, 1)


#%%
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

plt.figure()
plt.plot(X_test.flatten(), label='package')
plt.plot(Y_test.flatten(), label='reference', alpha=.8)
# plt.plot(Y_test_pred.flatten(), label='model')
plt.legend()
#%%
train_names = ["0-1 Hz Train", "1-2 Hz Train", "2-3 Hz Train", "3-4 Hz Train", "4-5 Hz Train",\
               "0-10 Hz Train"]
for i, name in enumerate(train_names):
    y_pred = Y_train_pred[i]
    y_train = Y_train[i]
    x_train = X_train[i]
    
    x = np.append(x_train, y_train, -1)
    x = np.append(x, y_pred, -1)
    
    np.savetxt("./model_predictions/%s.csv"%name, x, delimiter=',')

y_pred = Y_test_pred[0]
y_test = Y_test[0]
x_test = X_test[0]

x = np.append(x_test, y_test, -1)
x = np.append(x, y_pred, -1)

np.savetxt("./model_predictions/0-10 Hz Test.csv", x, delimiter=',')
#%% analysis on testing dataset results
pkg_rmse = mean_squared_error(y_test, x_test, squared=False)
model_rmse = mean_squared_error(y_test, y_pred, squared=False)
pkg_snr = signaltonoise(y_test, x_test)
model_snr = signaltonoise(y_test, y_pred)


# break by hertz range
y_pred = y_pred[:y_pred.size//10*10].reshape(10, -1)
y_test = y_test[:y_test.size//10*10].reshape(10, -1)
x_test = x_test[:x_test.size//10*10].reshape(10, -1)


pkg_rmse = mean_squared_error(y_test[:5], x_test[:5], squared=False)
model_rmse = mean_squared_error(y_test[:5], y_pred[:5], squared=False)
pkg_snr = signaltonoise(y_test[:5], x_test[:5])
model_snr = signaltonoise(y_test[:5], y_pred[:5])


results = np.zeros((10, 6))

for i, y_p, y_t, x_t in zip(range(10), y_pred, y_test, x_test):
    pkg_rmse = mean_squared_error(y_t, x_t, squared=False)
    pkg_snr = signaltonoise(y_t, x_t)
    
    model_rmse = mean_squared_error(y_t, y_p, squared=False)
    model_snr = signaltonoise(y_t, y_p)
    
    results[i] = [pkg_rmse, model_rmse, 100*(pkg_rmse-model_rmse)/pkg_rmse, pkg_snr, model_snr, 100*(model_snr-pkg_snr)/pkg_snr]