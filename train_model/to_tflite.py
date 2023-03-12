#%%
import numpy as np
from tensorflow.python import keras
import tensorflow.lite as tflite

#%%
# dummy_input = np.ones((1, 20000, 1))
# dummy_input = np.random.rand(1, 20000, 1)
dummy_input = np.random.rand(1, 20, 1)

#%%
model = keras.models.load_model("./model_saves/rnn_model")
y = model.predict(dummy_input)

#%%
cvt = tflite.TFLiteConverter.from_keras_model(model)
# cvt.target_spec.supported_ops = [
#     tflite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#     tflite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
# ]
cvt.experimental_new_converter = True
tflite_model = cvt.convert()

#%%
with open('./model_saves/rnn_model/model.tflite', 'wb') as f:
  f.write(tflite_model)
