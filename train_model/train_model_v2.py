import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
"""
Trying stateful=True

Tensorflow 2.10.0
"""
#%% function definitions
""" signal to noise ratio """
def signaltonoise(sig, noisy_signal, dB=True):
    noise = sig - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(sig)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
""" root relative squared error """
def rootrelsqerror(sig, pred):
    error = sig - pred
    mean = np.mean(sig)
    num = np.sum(np.square(error))
    denom = np.sum(np.square(sig-mean))
    return np.sqrt(num/denom)

""" training generator splits up dataset by train_len """
class TrainingGenerator(keras.utils.Sequence):
    
    def __init__(self, *args, train_len=400):
        self.args = args
        self.train_len = train_len
        self.length = args[0].shape[1]//train_len
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        rtrn = [arg[:,index*self.train_len:(index+1)*self.train_len,:] for arg in self.args]
        return rtrn[:-1], rtrn[-1] 

#%% load test and train data
X_test = np.load("./dataset/X_test.npy")
Y_test = np.load("./dataset/Y_test.npy")
X_train = np.load("./dataset/X_train.npy")
Y_train = np.load("./dataset/Y_train.npy")

# only go up to 5 Hz
X_train = np.append(X_train[:1], X_train[2:5], axis=0)
Y_train = np.append(Y_train[:1], Y_train[2:5], axis=0)
X_test = X_test[:2]
Y_test = Y_test[:2]

X_test = np.expand_dims(X_test, -1)
Y_test = np.expand_dims(Y_test, -1)
X_train = np.expand_dims(X_train, -1)
Y_train = np.expand_dims(Y_train, -1)

t_test = np.array([1/400*i for i in range(X_test.shape[1])])
t_train = np.array([1/400*i for i in range(X_train.shape[1])])

training_generator = TrainingGenerator(X_train, Y_train, train_len=400)
#%% create LSTM models and callbacks
UNITS = 30

model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=True, 
                      batch_input_shape=(X_train.shape[0], None, 1), 
                      trainable=True),
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=True, 
                      trainable=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, use_bias=True, trainable=True))
])
adam = keras.optimizers.Adam(
    learning_rate=0.001,
)
model.compile(
    loss='mse',
    optimizer=adam,
)
checkpoint_filepath = "./checkpoints/"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    # save_best_only=True,
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    # min_delta=0.001,
    patience=4,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    # start_from_epoch=15
)
#%% train model
print("beginning training...")
start_train_time = time.perf_counter()
model.fit(
    training_generator,
    shuffle=False,
    epochs=50,
    # validation_data = (X_test, Y_test),
    callbacks=[checkpoint],
)
stop_train_time = time.perf_counter()
elapsed_time = stop_train_time - start_train_time
print("training required %d minutes, %d seconds"%((int(elapsed_time/60)), int(elapsed_time%60)))
# model.load_weights(checkpoint_filepath) # restore best weights
model.save("./model_saves/model")
#%% remake model
# model = keras.models.load_model("./model_saves/model")
new_model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=False, 
                      input_shape=(None, 1), 
                      trainable=True),
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=False, 
                      trainable=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, use_bias=True, trainable=True))
])
for l1, l2 in zip(model.layers, new_model.layers):
    l2.set_weights(l1.get_weights())

model = new_model
#%% validation using RMSE and SNR
plt.close('all')
Y_pred = model.predict(X_test)

for x_test, y_test, y_pred in zip(X_test, Y_test, Y_pred):
    pkg_snr = signaltonoise(y_test, x_test)
    pkg_rmse = mean_squared_error(y_test, x_test, squared=False)
    model_snr = signaltonoise(y_test, y_pred)
    model_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("package accelerometer: ")
    print("SNR: %f"%pkg_snr)
    print("RMSE: %f"%pkg_rmse)
    print("with model correction: ")
    print("SNR: %f dB"%model_snr)
    print("RMSE: %f"%model_rmse)
    
    plt.figure(figsize=(6, 2.5))
    plt.plot(t_test, x_test, label='package')
    plt.plot(t_test, y_pred, label='prediction')
    plt.plot(t_test, y_test, label='reference')
    plt.plot()
    plt.legend(loc=1)
    plt.tight_layout()
#%% use first half of 0-10 Hz as validation
X_test_10 = np.load("./dataset/X_test.npy")[2]
Y_test_10 = np.load("./dataset/Y_test.npy")[2]
X_test_5 = X_test_10[:X_test_10.size//2].reshape(1, -1, 1)
Y_test_5 = Y_test_10[:Y_test_10.size//2].reshape(1, -1, 1)

y_pred = model.predict(Y_test_5).reshape(-1, 1)
y_test = Y_test_5.reshape(-1, 1)
x_test = X_test_5.reshape(-1, 1)

pkg_snr = signaltonoise(y_test, x_test)
pkg_rmse = mean_squared_error(y_test, x_test, squared=False)
model_snr = signaltonoise(y_test, y_pred)
model_rmse = mean_squared_error(y_test, y_pred, squared=False)
print("package accelerometer: ")
print("SNR: %f"%pkg_snr)
print("RMSE: %f"%pkg_rmse)
print("with model correction: ")
print("SNR: %f dB"%model_snr)
print("RMSE: %f"%model_rmse)

plt.figure(figsize=(6, 2.5))
plt.plot(t_test[:x_test.size], x_test, label='package')
plt.plot(t_test[:x_test.size], y_pred, label='prediction')
plt.plot(t_test[:x_test.size], y_test, label='reference')
plt.plot()
plt.legend(loc=1)
plt.tight_layout()
#%% prediction on training data
plt.close('all')
Y_pred = model.predict(X_train)

for x_train, y_train, y_pred in zip(X_train, Y_train, Y_pred):
    pkg_snr = signaltonoise(y_train, x_train)
    pkg_rmse = mean_squared_error(y_train, x_train, squared=False)
    model_snr = signaltonoise(y_train, y_pred)
    model_rmse = mean_squared_error(y_train, y_pred, squared=False)
    print("package accelerometer: ")
    print("SNR: %f"%pkg_snr)
    print("RMSE: %f"%pkg_rmse)
    print("with model correction: ")
    print("SNR: %f dB"%model_snr)
    print("RMSE: %f"%model_rmse)
    
    plt.figure(figsize=(6, 2.5))
    plt.plot(t_train, x_train, label='package')
    plt.plot(t_train, y_pred, label='prediction')
    plt.plot(t_train, y_train, label='reference')
    plt.plot()
    plt.legend(loc=1)
    plt.tight_layout()