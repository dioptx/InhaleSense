import sys
import os
# Relative import enabling
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from src.processing import make_tf_dataset, serve_single_file
from src.preparation import fetch_single_file
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import load_model
from scipy.io.wavfile import read
import pandas as pd
import numpy as np
import cufflinks
import math
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


model = load_model("model_100.h5")


def make_prediction(model):
    #Todo: make prediction pipeline
    pass