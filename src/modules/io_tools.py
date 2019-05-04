from scipy.io import wavfile
import pandas as pd
import os, csv, math
import seaborn as sns
from pathlib import Path
from numpy import genfromtxt

import matplotlib.pyplot as plot
import matplotlib.pylab as plt
from src.modules.visualizer import do_heatmap
from pprint import pprint
script_path = Path(os.path.realpath(__file__))
project_path = script_path.parent.parent.parent

dataset_folder = project_path / Path('assets/Whole')
pickle_folder = project_path / Path('assets/Pickles')




def create_dataset(target_path):
    '''
    This function creates the dataset for a given path of inputs
    :param path:
    :return: dataset and annotation
    '''
    annotations = pd.read_csv( target_path,
                              quotechar='"', skipinitialspace=True, names=['Filename', 'Label', 'begin', 'end'])

    filenames = set([annotation[0] for idx, annotation in annotations.iterrows()])

    dataset = pd.DataFrame(columns=['Filename', 'Label', 'Begin', 'End', 'Sample', 'CEPST','SPECT', 'MFCC', 'CWT', 'ZCR'])

    for idx, filename in enumerate(filenames):
        # Read the .wav file
        fs, data = wavfile.read(dataset_folder / filename)

        main_name = Path(filename.split('.')[0])

        # Read the feature files
        dat_cepst = genfromtxt( str(dataset_folder / main_name) + '_cepst.csv', delimiter=',')
        dat_spect = genfromtxt( str(dataset_folder / main_name) + '_spect.csv', delimiter=',')
        dat_mfcc = genfromtxt( str(dataset_folder / main_name) + '_mfcc.csv', delimiter=',')
        dat_cwt = genfromtxt( str(dataset_folder / main_name) + '_cwt.csv', delimiter=',')
        dat_zcr = genfromtxt( str(dataset_folder / main_name) + '_zcr.csv', delimiter=',')


        # Get the relevant entries from the dataset
        relevant = annotations.loc[annotations['Filename'] == filename]

        # Extract and append the entries with the audio snippet
        for idx, snippet in relevant.iterrows():
            begin, end = (snippet[2], snippet[3])

            q_begin, q_end = (math.floor(begin/128) , math.ceil( end/128) )

            # # Verification
            # print('Label: ',snippet[1], ' Length: ', len(data), ' Quantas 778 : ', len(data)/778,
            #       ' Quantas 128 : ', len(data)/128, ' Quant: {0} - {1}'.format(math.floor(begin/128),math.ceil( end/128)) )

            

            row_dict = {'Filename': snippet[0], 'Label': snippet[1], 'Begin': begin, 'End': end,
                        'Sample': [data[begin:end]],
                        'CEPST': [dat_cepst[: ,q_begin: q_end]],
                        'SPECT': [dat_spect[:, q_begin: q_end]],
                        'MFCC': [dat_mfcc[:, q_begin: q_end]],
                        'CWT': [dat_cwt[:, q_begin: q_end]],
                        'ZCR': [dat_zcr[ q_begin: q_end]]
                        }
            dataset.loc[len(dataset)] = row_dict

        print(idx)

    return annotations, dataset



# annotations, dataset = create_dataset(dataset_folder/ 'annotation_whole_dei.csv')
#
# annotations.to_pickle(os.path.join(pickle_folder, 'annotation_whole.pkl'))  # where to save it, usually as a .pkl
# dataset.to_pickle(os.path.join(pickle_folder, 'dataset_whole.pkl'))

# How to read

def fetch_dataset():
    annotations = pd.read_pickle(pickle_folder / 'annotation_whole.pkl')
    dataset = pd.read_pickle(pickle_folder / 'dataset_whole.pkl')

    return annotations, dataset

