from tensorflow import python as tf

from src.processing import make_dataset, serve_single_file


from src.modules.visualizer import do_heatmap
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
import tensorflow

# Processing unit verbosity
# tensorflow.debugging.set_log_device_placement(True)


# This is the size of the window that is fed into the DNN
window_size = 15
# The number of the features present in the dataset
num_of_features = 42
# Number of distinct labels in the output
label_length = 4
# Hyperparameter that defines the number of samples to work through
# before updating the internal model parameters.
batch_size = 25





# Create a model
model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(window_size, num_of_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(label_length, activation='softmax'))
model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

data_train = make_dataset(window_size, num_of_features, label_length, test=False).batch(batch_size)
data_test = make_dataset(window_size, num_of_features, label_length, test=True).batch(batch_size)



model.fit(data_train, epochs=1)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(data_test)

print('\ntest loss, test acc: \n', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(data_test)

print('predictions:', predictions)

dataset_test = serve_single_file('rec2018-01-22_17h41m33.475s.wav', 15, 42, 4, test= False).batch(batch_size)
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(dataset_test)



#TODO: Make: Wav to predict, Noise extraction - Dataset Improvement
