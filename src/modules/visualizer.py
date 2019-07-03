from pprint import pprint
import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib.pylab as plt
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode


def do_heatmap(input, title='', xlabel='time/128bits', ylabel='frequencies'):
    '''
    Creates a heatmap for a 2d array
    :param input:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    '''
    # plt.figure(figsize=(16, 6))

    sns.heatmap(input)

    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.show()



def plot_wav(wav_file):
    pass