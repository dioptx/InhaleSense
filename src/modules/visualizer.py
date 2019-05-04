from pprint import pprint
import csv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plot
import matplotlib.pylab as plt



def do_heatmap(input, title='', xlabel='time/128bits', ylabel='frequencies'):
    '''
    Creates a heatmap for a 2d array
    :param input:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    '''
    plot.subplot()
    sns.heatmap(input)
    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.show()

