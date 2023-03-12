## LSTM Compensator
This folder contains all the files used to create the LSTM compensator models.

### Dataset
Properly formatted data from the accelerometer experiments is shared in [/dataset](LSTM%20compensator/dataset). Preprocessing included aligning the two signals and sampling to a consistent rate, and selecting training and testing experiments. The files are given in numpy `.npy` files. "X" datasets are input data from the sensor package and "Y" files are the reference signals from the reference accelerometer.

### Python files
The most recent training file is in `train_model_v4.py` which uses the V4 version of the training dataset. The model is saved to [/model_saves](LSTM%20compensator/model_saves) in the TensorFlow SavedModel format. Model predictions on both the training and testing datasets are also saved in [/model_predictions](LSTM%20compensator/model_predictions) as .csv files.

### Dependencies

tensorflow 2.10.0\
numpy 1.23.3
