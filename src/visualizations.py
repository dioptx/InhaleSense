import os
import sys

from src.processing import serve_single_file
from tensorflow.python.keras.models import load_model
from scipy.io.wavfile import read
import pandas as pd
import numpy as np
from plotly import tools
import matplotlib.pyplot as plt
import cufflinks
import math
from itertools import repeat

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode

# Relative import enabling
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# Using plotly + cufflinks in offline mode


# function below sets the color based on amount

def SetColor(x, ph):
    if math.floor(x / 128) < len(ph):
        p = ph[math.floor(x / 128)]
    else:
        return "black"
    if p == [1, 0, 0, 0]:
        return "blue"

    if p == [0, 1, 0, 0]:
        return "red"

    if p == [0, 0, 1, 0]:
        return "green"

    return "black"


def visualize_prediction(file_path: str, prediction_list):
    (fs, x) = read('/Users/noresources/Pycharm_projects/InhaleSense/data/raw/' + file_path)

    # Create a trace
    predicted = go.Scattergl(
        x=np.arange(0, x.shape[0]),
        y=x,
        marker=dict(
            color=list(map(SetColor, np.arange(0, x.shape[0]), repeat(prediction_list))),
            size=1.5,
            opacity=0.5),
        mode="markers"
    )

    return predicted


def visualize_groundtruth(file_path: str, relevant):
    (fs, x) = read('/Users/noresources/Pycharm_projects/InhaleSense/data/raw/' + file_path)

    Exhale = []
    Inhale = []
    Drug = []
    for idx, snippet in relevant.iterrows():
        if snippet[1] == 'Exhale':
            Exhale.append({'begin': snippet[2], 'end': snippet[3]})
        elif snippet[1] == 'Inhale':
            Inhale.append({'begin': snippet[2], 'end': snippet[3]})
        elif snippet[1] == 'Drug':
            Drug.append({'begin': snippet[2], 'end': snippet[3]})

    # print(Exhale, Inhale, Drug)

    # function below sets the color based on amount
    def SetColor_range(x):
        for exh in Exhale:
            if exh['begin'] < x and exh['end'] > x:
                return "red"

        for inh in Inhale:
            if inh['begin'] < x and inh['end'] > x:
                return "blue"

        for dru in Drug:
            if dru['begin'] < x and dru['end'] > x:
                return "green"

        return "black"

    # Create a trace
    ground_truth = go.Scattergl(
        x=np.arange(0, x.shape[0]),
        y=x,
        marker=dict(
            color=list(map(SetColor_range, np.arange(0, x.shape[0]))),
            size=1.5,
            opacity=0.5),
        mode="markers"
    )

    return ground_truth


def visualize_model_training(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# This is the size of the window that is fed into the DNN
window_size = 15
# The number of the features present in the dataset
num_of_features = 42
# Number of distinct labels in the output
label_length = 4
# Hyperparameter that defines the number of samples to work through
# before updating the internal model parameters.
batch_size = 25
YOUR_WAV_PATH = 'rec2018-01-22_17h41m33.475s.wav'


def evaluate_model(filepath: str, visualize: bool):
    # Replace with your target path
    YOUR_ANNOTATION_PATH = '/Users/noresources/Pycharm_projects/InhaleSense/data/raw/annotation_whole_dei.csv'

    annotations = pd.read_csv(YOUR_ANNOTATION_PATH,
                              quotechar='"', skipinitialspace=True, names=['Filename', 'Label', 'begin', 'end'])
    # Get the relevant entries from the dataset
    relevant = annotations.loc[annotations['Filename'] == filepath]

    dataset_test = serve_single_file(filepath,
                                     window_size, num_of_features, label_length, test=False).batch(batch_size)

    model = load_model("/Users/noresources/Pycharm_projects/InhaleSense/notebook/model_100.h5")

    predictions = model.predict(dataset_test)

    ph = []

    for pred in predictions:
        i = list(pred).index(max(list(pred)))
        if i == 0:
            ph.append([1, 0, 0, 0])
        elif i == 1:
            ph.append([0, 1, 0, 0])
        elif i == 2:
            ph.append([0, 0, 1, 0])
        else:
            ph.append([0, 0, 0, 1])

    if visualize:
        (fs, x) = read('/Users/noresources/Pycharm_projects/InhaleSense/data/raw/' + YOUR_WAV_PATH)

        # Create a trace
        initial = go.Scattergl(
            x=np.arange(0, x.shape[0]),
            y=x,
            marker=dict(
                color='orange',
                size=1.5,
                opacity=0.5),
            mode="markers"
        )

        gt = visualize_groundtruth(YOUR_WAV_PATH, relevant)

        pred = visualize_prediction(YOUR_WAV_PATH, prediction_list=ph)

        fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('Original', 'Ground Truth', 'Predicted'))

        fig.append_trace(initial, 1, 1)
        fig.append_trace(gt, 2, 1)
        fig.append_trace(pred, 3, 1)

        a_plot = plot(fig)


# evaluate_model(YOUR_WAV_PATH, True)
