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

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy.io import wavfile

script_path = Path(os.path.realpath(__file__))
project_path = script_path.parent.parent

raw_folder = project_path / Path('data/raw')
processed_folder = project_path / Path('data/processed')


def make_label(timepoint: int, query: pd.DataFrame):
    '''
    Finds the correct label for a set timepoint
    :param timepoint: the index of the timepoint
    :param query: the query holding the labels
    :return:
    '''
    for idx, row in query.iterrows():
        if row['Begin'] >= timepoint and timepoint <= row['End']:
            return hash_label(row['Label'])

    return hash_label('Noise')


def hash_label(label_name: str, make_int=False):
    '''
    Hashes the label name into an array
    :param label_name:
    :return:
    '''
    if label_name == 'Inhale':
        if make_int:
            label = 0
        else:
            label = [1, 0, 0, 0]
    elif label_name == 'Exhale':
        if make_int:
            label = 1
        else:
            label = [0, 1, 0, 0]
    elif label_name == 'Drug':
        if make_int:
            label = 2
        else:
            label = [0, 0, 1, 0]
    else:
        if make_int:
            label = 3
        else:
            label = [0, 0, 0, 1]

    return label


def create_dataset_slim(target_path: str, target_name: str):
    '''
    This function creates the dataset for a given path of inputs
    :param path:
    :return: dataset and annotation
    '''
    annotations = pd.read_csv(target_path,
                              quotechar='"', skipinitialspace=True, names=['Filename', 'Label', 'begin', 'end'])

    filenames = set([annotation[0] for idx, annotation in annotations.iterrows()])

    dataset = pd.DataFrame(
        columns=['Filename', 'LabelVector', 'Begin', 'End', 'Sample', 'CEPST', 'SPECT', 'MFCC', 'CWT', 'ZCR'])

    for i, filename in enumerate(filenames):
        # Read the .wav file
        fs, data = wavfile.read(raw_folder / filename)

        main_name = Path(filename.split('.')[0])

        # Read the feature files
        dat_cepst = genfromtxt(str(raw_folder / main_name) + '_cepst.csv', delimiter=',')
        dat_spect = genfromtxt(str(raw_folder / main_name) + '_spect.csv', delimiter=',')
        dat_mfcc = genfromtxt(str(raw_folder / main_name) + '_mfcc.csv', delimiter=',')
        dat_cwt = genfromtxt(str(raw_folder / main_name) + '_cwt.csv', delimiter=',')
        dat_zcr = genfromtxt(str(raw_folder / main_name) + '_zcr.csv', delimiter=',')

        # Get the relevant entries from the dataset
        relevant = annotations.loc[annotations['Filename'] == filename]

        # Extract and append the entries with the audio snippet
        for idx, snippet in relevant.iterrows():
            begin, end = (snippet[2], snippet[3])

            q_begin, q_end = (math.floor(begin / 128), math.ceil(end / 128))

            # # Verification
            # print('Label: ',snippet[1], ' Length: ', len(data), ' Quantas 778 : ', len(data)/778,
            #       ' Quantas 128 : ', len(data)/128, ' Quant: {0} - {1}'.format(math.floor(begin/128),math.ceil( end/128)) )

            row_dict = {'Filename': snippet[0], 'LabelVector': snippet[1], 'Begin': begin, 'End': end,
                        'Sample': [data[begin:end]],
                        'CEPST': [dat_cepst[:, q_begin: q_end]],
                        'SPECT': [dat_spect[:, q_begin: q_end]],
                        'MFCC': [dat_mfcc[:, q_begin: q_end]],
                        'CWT': [dat_cwt[:, q_begin: q_end]],
                        'ZCR': [dat_zcr[q_begin: q_end]]
                        }
            dataset.loc[len(dataset)] = row_dict

        print(i)

    dataset.to_pickle(os.path.join(processed_folder, target_name + '.pkl'))

    return annotations, dataset


def create_dataset(target_path: Path, target_name: str):
    annotations = pd.read_csv(target_path,
                              quotechar='"', skipinitialspace=True, names=['Filename', 'Label', 'begin', 'end'])

    filenames = set([annotation[0] for idx, annotation in annotations.iterrows()])

    dataset = pd.DataFrame(
        columns=['Filename', 'LabelVector', 'Sample', 'CEPST', 'SPECT', 'MFCC', 'CWT', 'ZCR'])

    for idx, filename in enumerate(filenames):
        # Read the .wav file
        fs, data = wavfile.read(raw_folder / filename)

        main_name = Path(filename.split('.')[0])
        # Read the feature files
        dat_cepst = genfromtxt(str(raw_folder / main_name) + '_cepst.csv', delimiter=',')
        dat_spect = genfromtxt(str(raw_folder / main_name) + '_spect.csv', delimiter=',')
        dat_mfcc = genfromtxt(str(raw_folder / main_name) + '_mfcc.csv', delimiter=',')
        dat_cwt = genfromtxt(str(raw_folder / main_name) + '_cwt.csv', delimiter=',')
        dat_zcr = genfromtxt(str(raw_folder / main_name) + '_zcr.csv', delimiter=',')

        # Get the relevant entries from the dataset
        relevant = annotations.loc[annotations['Filename'] == filename]
        # Figure out the sample window size that was chosen when creating the features
        sample_window_size = len(data) / dat_spect.shape[1]

        # Create a matrix initialised with the Noise Label
        label_vector = np.zeros((4, int(len(data) / sample_window_size)), int)
        label_vector[3, :] = 1

        # Extract and append the entries with the audio snippet
        for index, snippet in relevant.iterrows():
            begin, end = (math.floor(snippet[2] / sample_window_size), math.ceil(snippet[3] / sample_window_size))
            # Make the label patch
            patch = np.zeros((4, end - begin), int)
            h = hash_label(snippet[1]).index(1)
            patch[h, :] = 1

            # Patch the label_vector
            label_vector[:, begin: end] = patch

        # Complete the row
        row_dict = {'Filename': filename, 'LabelVector': [label_vector],
                    'Sample': [data],
                    'CEPST': [dat_cepst],
                    'SPECT': [dat_spect],
                    'MFCC': [dat_mfcc],
                    'CWT': [dat_cwt],
                    'ZCR': [dat_zcr]
                    }
        # Print some statistics
        print(idx, sample_window_size, len(label_vector[0]), fs, filename)

        dataset.loc[len(dataset)] = row_dict

    dataset.to_pickle(os.path.join(processed_folder, target_name + '.pkl'))


def fetch_dataset(target_name: str):
    # annotations = pd.read_pickle(processed_folder / target_name)
    dataset = pd.read_pickle(processed_folder / target_name)

    return dataset


def fetch_single_file(filename: str):
    annotations = pd.read_pickle(processed_folder / 'annotation_whole.pkl')
    dataset = pd.read_pickle(processed_folder / 'dataset_whole.pkl')

    # Get the relevant rows of the dataset
    query = dataset.loc[dataset['Filename'] == filename]

    main_name = Path(filename.split('.')[0])
    image_dataset = genfromtxt(str(raw_folder / main_name) + '_spect.csv', delimiter=',')
    # do_heatmap(image_dataset)

    data_array = []
    data_labels = []

    for idx, sublist in enumerate(image_dataset.T):
        # Append the features on the data_array
        data_array.append(sublist)
        # Index multiplied by the sampling factor
        timepoint = idx * 128
        data_labels.append(make_label(timepoint, query))

    if len(data_labels) != len(data_array):
        raise Exception('The length of data_array:{0} and data_labels:{1} should be equal.'.format(len(data_array),
                                                                                                   len(data_labels)))

    return data_array, data_labels
