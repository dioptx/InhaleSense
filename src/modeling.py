from tensorflow import python as tf

from src.processing import make_tf_dataset, split_train_test, fetch_dataset

from src.visualizations import visualize_model_training
from src.modules.visualizer import do_heatmap
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
from sklearn.metrics import confusion_matrix
from src.config import LstmConfig
from src.processing import dataset_to_array, make_tf_dataset

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



lconf = LstmConfig()


class inhalerPredictor():

    def __init__(self, hparams):
        self.lconf = hparams

        self.model = self.build_model()

    def build_model(self):

        lconf = self.lconf
        # Create a model
        model = Sequential()

        model.add(LSTM(512, return_sequences=False, input_shape=(lconf.window_size, lconf.num_of_features)))
        model.add(Dropout(lconf.dropout))
        model.add(Flatten())
        model.add(Dense(lconf.label_length, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer=lconf.optimizer,
                      metrics=['accuracy'])

        return model



def list_to_num(el):
    if el == [1, 0, 0, 0]:
        return 0
    elif el == [0, 1, 0, 0]:
        return 1
    elif el == [0, 0, 1, 0]:
        return 2
    elif el == [0, 0, 0, 1]:
        return 3
    else:
        print('Error', el)


def test_model(md, dt):
    print('\n')
    data_test = make_tf_dataset(dt, window_size, num_of_features, label_length).batch(batch_size)
    results = md.evaluate(data_test)

    ph = []
    predictions = md.predict(data_test)
    for pred in predictions:
        i = list(pred).index(max(list(pred)))
        ph.append(i)
    # Ground truth
    fm, labels = dataset_to_array(dt)
    # Padding
    ph = ph + [3 for i in range(0, len(labels) - len(ph))]

    # -----------------
    labels = [list_to_num(list(x)) for x in labels]
    # CM
    cm = confusion_matrix(labels, ph)
    print('\n', cm)