import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
"""
Changing to V4 dataset
Training:
    0-1 Hz
    2-3 Hz
    3-4 Hz
    4-5 Hz
    5-6 Hz
    6-7 Hz
    7-8 Hz
    8-9 Hz
    9-10 Hz
Testing on:
    0-10 Hz portion of 0-20 Hz

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

def save_model_weights_as_csv(model, savpath = "./model_weights"):
    import os
    import os.path as path
    from numpy import savetxt
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    for layer in model.layers[:-1]: # the final layer is a dense top
        layer_path = savpath + "./" + layer.name + "./"
        if(not path.exists(layer_path)):
            os.mkdir(layer_path)
        W, U, b = layer.get_weights()
        units = U.shape[0]
        
        savetxt(layer_path+"Wi.csv",W[:,:units].T,delimiter=',')
        savetxt(layer_path+"Wf.csv",W[:,units:units*2].T,delimiter=',')
        savetxt(layer_path+"Wc.csv",W[:,units*2:units*3].T,delimiter=',')
        savetxt(layer_path+"Wo.csv",W[:,units*3:].T,delimiter=',')
        savetxt(layer_path+"Ui.csv",U[:,:units].T,delimiter=',')
        savetxt(layer_path+"Uf.csv",U[:,units:units*2].T,delimiter=',')
        savetxt(layer_path+"Uc.csv",U[:,units*2:units*3].T,delimiter=',')
        savetxt(layer_path+"Uo.csv",U[:,units*3:].T,delimiter=',')
        savetxt(layer_path+"bi.csv",b[:units],delimiter=',')
        savetxt(layer_path+"bf.csv",b[units:units*2],delimiter=',')
        savetxt(layer_path+"bc.csv",b[units*2:units*3],delimiter=',')
        savetxt(layer_path+"bo.csv",b[units*3:],delimiter=',')
    
    #save dense top layer
    dense_top = model.layers[-1]
    in_weights, out_weights = dense_top.get_weights()
    layer_path = savpath + "./dense_top./"
    if(not path.exists(layer_path)):
        os.mkdir(layer_path)    
    savetxt(layer_path+"weights.csv",in_weights,delimiter=',')
    savetxt(layer_path+"bias.csv",out_weights,delimiter=',')
#%% load test and train data

X_train = np.load("./dataset/V4/X_train.npy")
Y_train = np.load("./dataset/V4/Y_train.npy")
X_test = np.load("./dataset/V4/X_test.npy").reshape(1, -1, 1)
Y_test = np.load("./dataset/V4/Y_test.npy").reshape(1, -1, 1)

fs = X_train.shape[1]

X_train = X_train[:,:fs//2].reshape(10, -1, 1)
Y_train = Y_train[:,:fs//2].reshape(10, -1, 1)


t_train = np.array([1/400*i for i in range(X_train.shape[1])])
t_test = np.array([1/400*i for i in range(X_test.shape[1])])


training_generator = TrainingGenerator(X_train, Y_train, train_len=400)
#%% create LSTM models and callbacks
UNITS = 50

model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=False, 
                      batch_input_shape=(X_train.shape[0], None, 1), 
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
    shuffle=True,
    epochs=70,
    # validation_data = (X_test, Y_test),
    callbacks=[checkpoint],
)
stop_train_time = time.perf_counter()
elapsed_time = stop_train_time - start_train_time
print("training required %d minutes, %d seconds"%((int(elapsed_time/60)), int(elapsed_time%60)))
# model.load_weights(checkpoint_filepath) # restore best weights
model.save("./model_saves/model")
save_model_weights_as_csv(model)
#%% remake model
# model = keras.models.load_model("./model_saves/model")
new_model = keras.Sequential([
    keras.layers.LSTM(UNITS, 
                      return_sequences=True, 
                      stateful=False, 
                      input_shape=(None, 1), 
                      trainable=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1, use_bias=True, trainable=True))
])
for l1, l2 in zip(model.layers, new_model.layers):
    l2.set_weights(l1.get_weights())
model = new_model
model.save("./model_saves/model")
#%% validation using RMSE and SNR, use first half of 0-20 Hz as validation
plt.close('all')

# y_pred = np.load("./model_predictions/Y_pred.npy")
y_pred = model.predict(X_test).reshape(-1, 1)
y_test = Y_test.reshape(-1, 1)
x_test = X_test.reshape(-1, 1)

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

# np.save("./model_predictions/Y_pred.npy", y_pred)
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