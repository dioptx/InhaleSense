from tensorflow import python as tf
from src.preparation import fetch_dataset, fetch_single_file, hash_label
from src.modules.visualizer import do_heatmap
import numpy as np
import pandas as pd
import math
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM
import tensorflow

def split_train_test_val(dataset):
    total = len(dataset)
    a, b = (math.ceil(total*0.7), math.ceil(total*0.85) )
    train, test, val = (dataset[0:a], dataset[a+1: b], dataset[b+1:])
    return train, test, val

def split_train_test(dataset, percentage: float):
    dt_size = len(dataset)
    mark = math.floor(dt_size * percentage)

    return dataset[:mark], dataset[mark + 1:]




def dataset_to_array(dataset: pd.DataFrame):
    feature_matrix = []
    label_matrix = []

    for idx, row in dataset.iterrows():

        # Extract the feature and the labels
        spect = row['SPECT'][0]
        if type(row['LabelVector']) == str:
            h = hash_label(row['LabelVector']).index(1)

            # Create a matrix from the Label
            label_vector = np.zeros((4, spect.shape[1]), int)
            label_vector[h, :] = 1


        # print(len(feature_matrix[0][0]))
        # print(label_matrix[0].shape)

        if not (spect.shape[1] == label_vector.shape[1]):
            raise Exception('Padding required, feature and label vectors not equal length.'
                            'Lengths:', spect.shape[1], label_vector.shape[1])

        # Append the feature and the labels to the dataset
        for idx in range(spect.shape[1]):
            feature_matrix.append(spect[:, idx])
            label_matrix.append(label_vector[:, idx])

    return np.array(feature_matrix), np.array(label_matrix)


def dataset_slim_to_array(dataset: pd.DataFrame):
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


def dataset_to_generator_depr(window_size: int, dataset: list, labels: list, test):
    dt_size = len(dataset)
    mark = math.floor(dt_size * 0.8)

    if test:
        print('Dataset created with {0} entries'.format(dt_size - mark))

        for idx, feature in enumerate(dataset[mark:]):
            window = dataset[idx: idx + window_size]
            if len(window) == window_size:
                yield np.array(window), np.array(labels[idx])
    else:
        print('Dataset created with {0} entries'.format(mark))

        for idx, feature in enumerate(dataset[0:mark]):
            window = dataset[idx: idx + window_size]
            if len(window) == window_size:
                yield np.array(window), np.array(labels[idx])


def dataset_to_generator(window_size: int, dataset: list, labels: list):
    for idx, feature in enumerate(dataset):
        window = dataset[idx: idx + window_size]
        if len(window) == window_size:
            yield np.array(window), np.array(labels[idx])


def make_tf_dataset(dataset: pd.DataFrame, window_size: int, num_of_features: int, label_length: int):
    data_array, data_labels = dataset_to_array(dataset=dataset)

    # data_labels = [np.where(label == 1) for label in data_labels]
    # label_length = 4

    generator = lambda: dataset_to_generator(window_size, data_array, data_labels)
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((window_size, num_of_features), [label_length,]))



def serve_single_file(filename: str, window_size: int, num_of_features: int, label_length: int):
    data_array, data_labels = fetch_single_file(filename)
    print(data_labels[0])
    print(data_array[0])
    generator = lambda: dataset_to_generator(window_size, data_array, data_labels)

    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((window_size, num_of_features), (label_length,)))



