from scipy.io import wavfile
import pandas as pd
import os

dataset_folder = 'INPUT_DATASET_FOLDER'
pickle_folder = 'OUTPUT_DATASET_FOLDER'


def create_dataset(path):
    annotations = pd.read_csv(path,
                              quotechar='"', skipinitialspace=True, names=['Filename', 'Label', 'begin', 'end'])

    filenames = set([annotation[0] for idx, annotation in annotations.iterrows()])

    dataset = pd.DataFrame(columns=['Filename', 'Label', 'Begin', 'End', 'Sample'])

    for idx, filename in enumerate(filenames):
        # Read the .wav file
        fs, data = wavfile.read(os.path.join(dataset_folder, filename))
        # Get the relevant entries from the dataset
        relevant = annotations.loc[annotations['Filename'] == filename]

        # Extract and append the entries with the audio snippet
        for idx, snippet in relevant.iterrows():
            begin, end = (snippet[2], snippet[3])
            row_dict = {'Filename': snippet[0], 'Label': snippet[1], 'Begin': begin, 'End': end,
                        'Sample': [data[begin:end]]}
            dataset.loc[len(dataset)] = row_dict

        print(idx)

    # Save the dataset to memory
    annotations.to_pickle(os.path.join(pickle_folder, 'annotation_whole.pkl'))  # where to save it, usually as a .pkl
    dataset.to_pickle(os.path.join(pickle_folder, 'dataset_whole.pkl'))

# How to read

# annotations = pd.read_pickle(os.path.join(pickle_folder,'annotation_f1.pkl'))
# dataset = pd.read_pickle(os.path.join(pickle_folder,'dataset_f1.pkl'))
#
# for idx, row in dataset.iterrows():
#     print(row[1],len(row[-1][0]))
