from tensorflow import python as tf

from src.processing import make_tf_dataset, serve_single_file, split_train_test, fetch_dataset

from src.visualizations import visualize_model_training
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
batch_size = 100

target_name = 'dataset_all_slim.pkl'


class inhalerLstm():

    def __init__(self):
        self.window_size = window_size
        self.num_of_features = num_of_features
        self.label_length = label_length
        self.batch_size = batch_size

        self.model = self.build_model(self.window_size, self.num_of_features, self.label_length)

    @staticmethod
    def build_model(window_size: int, num_of_features: int, label_length: int):
        # Create a model
        model = Sequential()
        model.add(LSTM(100, return_sequences=False, input_shape=(window_size, num_of_features)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(label_length, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    def fit(self, *argv):
        return self.model.fit(argv)

    def evaluate(self, *argv):
        return self.model.evaluate(argv)

    def predict(self, *argv):
        return self.model.predict(argv)


model = inhalerLstm()

dataset = fetch_dataset(target_name=target_name)

data_train, data_test = split_train_test(dataset=dataset, percentage=0.8)

data_train = make_tf_dataset(data_train, window_size, num_of_features, label_length).batch(batch_size)
data_test = make_tf_dataset(data_test, window_size, num_of_features, label_length).batch(batch_size)

# train your model
history = model.model.fit(data_train, validation_data=data_test, epochs=100)

visualize_model_training(history)
# Evaluate the model on the test data using `evaluate`
# print('\n# Evaluate on test data')
# results = model.evaluate(data_test)

# print('\ntest loss, test acc: \n', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
# print('\n# Generate predictions for 3 samples')
# predictions = model.predict(data_test)

# print('predictions:', predictions)
model.model.save('model.ckpt')

dataset_test = serve_single_file('rec2018-01-22_17h41m33.475s.wav', 15, 42, 4, test=False).batch(batch_size)
# Evaluate the model on the test data using `evaluate`

# TODO: Make: Wav to predict, Noise extraction - Dataset Improvement
