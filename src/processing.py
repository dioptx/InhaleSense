from tensorflow import python as tf
from src.preparation import fetch_dataset
from src.modules.visualizer import do_heatmap
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
import tensorflow




def dataset_to_array():
    annotations, dataset = fetch_dataset()

    # Metrics
    arr_Exh = []  # Exhale
    num_Exh = 0
    arr_Inh = []  # Inhale
    num_Inh = 0
    arr_Dru = []  # Drug
    num_Dru = 0
    arr_Noi = []  # Noise
    num_Noi = 0

    image_dataset = []
    label_dataset = []

    for idx, row in dataset.iterrows():
        if row['Label'] == 'Noise':
            arr_Noi.append(len(row['Sample'][0]))
            num_Noi += 1
            if len(row['Sample'][0]) == 0:
                a = row['Sample'][0]
                pass
        elif row['Label'] == 'Exhale':
            arr_Exh.append(len(row['Sample'][0]))
            num_Exh += 1

        elif row['Label'] == 'Inhale':
            arr_Inh.append(len(row['Sample'][0]))
            num_Inh += 1

        elif row['Label'] == 'Drug':
            arr_Dru.append(len(row['Sample'][0]))
            num_Dru += 1

        label = row['Label']
        if label == 'Inhale':
            label = [1, 0, 0, 0]
        elif label == 'Exhale':
            label = [0, 1, 0, 0]
        elif label == 'Drug':
            label = [0, 0, 1, 0]
        elif label == 'Noise':
            label = [0, 0, 0, 1]

        spect = row['SPECT'][0]

        image_dataset.append(spect)
        label_dataset.append([label for i in range(0, spect.shape[1])])

        label = row['Label']
        cepst = row['CEPST'][0]
        # do_heatmap(cepst, title='CEPST'+label )

        # spect = row['SPECT'][0]
        # do_heatmap(spect, title='SPECT ' + label)
        #
        # mfcc = row['MFCC'][0]
        # do_heatmap(mfcc, title='MFCC ' + label)
        #
        # cwt = row['CWT'][0]
        # do_heatmap(cwt, title='cwt ' + label)
        # wav = row['Sample']
        # do_heatmap(wav)

        # print(row['Label'])

    # Serialization

    data_array = []

    for sublist in image_dataset:
        for item in sublist.T:
            data_array.append(item)

    data_labels = []
    for sublist in label_dataset:
        for item in sublist:
            data_labels.append(item)

    # for idx, feature in enumerate(data_array):
    #     window = data_array[idx: idx + 15]
    #     if len(window) == 15:
    #         print(idx)
    #     else:
    #         print('Edge Boi')

    # Metrics
    # print('Exhale:\n MinL: {0}, MaxL: {1}, Num: '.format(min(arr_Exh), max(arr_Exh), 'Num: ', num_Exh))
    # print('Inhale:\n MinL: {0}, MaxL: {1}, Num: '.format(min(arr_Inh), max(arr_Inh), 'Num: ', num_Inh))
    # print('Drug:\n MinL: {0}, MaxL: {1}, Num: '.format(min(arr_Dru), max(arr_Dru), 'Num: ', num_Dru))
    # print('Noise:\n MinL: {0}, MaxL: {1}, Num: '.format(min(arr_Noi), max(arr_Noi), 'Num: ', num_Noi))

    return data_array, data_labels


def dataset_to_generator(window_size, dataset, labels, test):

    if test:
        for idx, feature in enumerate(dataset[10001:20000]):
            window = dataset[idx: idx + window_size]
            if len(window) == window_size:
                yield np.array(window), np.array(labels[idx])
    else:
        for idx, feature in enumerate(dataset[0:10000]):
            window = dataset[idx: idx + window_size]
            if len(window) == window_size:
                yield np.array(window), np.array(labels[idx])


def make_dataset(window_size, num_of_features, label_length, test=False):
    data_array, data_labels = dataset_to_array()

    generator = lambda: dataset_to_generator(window_size, data_array, data_labels, test)
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((window_size, num_of_features), (label_length,)))


