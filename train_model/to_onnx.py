#%%
import numpy as np
import onnxruntime as ort
from tensorflow.python import keras

#%%
# dummy_input = np.ones((1, 20000, 1))
# dummy_input = np.random.rand(1, 20000, 1)
dummy_input = np.random.rand(1, 20, 1)


#%%
model = keras.models.load_model("./model_saves/rnn_model")
y = model.predict(dummy_input)

#%%
model_ort = ort.InferenceSession("./model_saves/rnn_model/model.onnx")
y_ort = model_ort.run(
    ["simple_rnn_6"],
    {"lstm_6_input": dummy_input.astype(np.float32)},
)[0]

# %%
np.isclose(y, y_ort).all()

#%%
np.sum(np.abs(y-y_ort[0]))
