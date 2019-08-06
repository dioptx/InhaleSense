"""
MIT License

Copyright (c) 2019 Dionisis Pettas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import os
import sys

# Relative import enabling
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.config import LstmConfig
from src.modeling import inhalerPredictor
from src.processing import split_train_test, make_tf_dataset, fetch_dataset
from src.visualizations import visualize_model_training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Testing ground


#
# lconf = LstmConfig()
#
# predictor = inhalerPredictor()
#
# dataset = fetch_dataset(target_name=target_name)
#
# data_train, data_test = split_train_test(dataset=dataset, percentage=0.8)
#
# data_train = make_tf_dataset(data_train, window_size, num_of_features, label_length).batch(batch_size)
# data_test = make_tf_dataset(data_test, window_size, num_of_features, label_length).batch(batch_size)
#
# history = predictor.model.fit(data_train, validation_data=data_test, epochs=100)
#
# visualize_model_training(history)
#
# dataset_test = serve_single_file('rec2018-01-22_17h41m33.475s.wav', 15, 42, 4, test=False).batch(batch_size)
# # Evaluate the model on the test data using `evaluate`
