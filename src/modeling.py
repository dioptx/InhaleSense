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

from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential

from src.config import LstmConfig
from src.processing import dataset_to_array, make_tf_dataset, split_train_test_val

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
    data_test = make_tf_dataset(dt, lconf.window_size, lconf.num_of_features, lconf.label_length).batch(
        lconf.batch_size)
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


def build_and_test(model, dataset, name, epochs):
    train, test, val = split_train_test_val(dataset)

    data_train = make_tf_dataset(train, lconf.window_size, lconf.num_of_features, lconf.label_length).batch(
        lconf.batch_size)
    data_test = make_tf_dataset(test, lconf.window_size, lconf.num_of_features, lconf.label_length).batch(
        lconf.batch_size)

    model.fit(data_train, epochs=epochs)
    data_val = make_tf_dataset(val, lconf.window_size, lconf.num_of_features, lconf.label_length).batch(
        lconf.batch_size)

    results = model.evaluate(data_val)
    test_model(model, val)
    print(results)
    model.save('model_{0}_{1}.h5'.format(name, epochs))  # creates a HDF5 file 'my_model.h5'
