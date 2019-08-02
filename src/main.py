import sys
import os
# Relative import enabling
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM

from src.config import LstmConfig
from src.modeling import inhalerPredictor
from src.processing import split_train_test, make_tf_dataset, fetch_dataset
from src.visualizations import visualize_model_training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

lconf = LstmConfig()






predictor = inhalerPredictor()

dataset = fetch_dataset(target_name=target_name)

data_train, data_test = split_train_test(dataset=dataset, percentage=0.8)

data_train = make_tf_dataset(data_train, window_size, num_of_features, label_length).batch(batch_size)
data_test = make_tf_dataset(data_test, window_size, num_of_features, label_length).batch(batch_size)

history = predictor.model.fit(data_train, validation_data=data_test, epochs=100)

visualize_model_training(history)


dataset_test = serve_single_file('rec2018-01-22_17h41m33.475s.wav', 15, 42, 4, test=False).batch(batch_size)
# Evaluate the model on the test data using `evaluate`



