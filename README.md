# InhaleSense

The goal of this project is to produce a deep learning model that discovers and classifies sections of interest in audio files.


The dataset used for the initial stage of the implementation is not provided. 


### Some other notes:


- The approach makes use of a simple LSTM model in order to discover inhalations, exhalations and Drug administration in 
.wav audio files.

- A trained model can be found in the data section.

- The preparation and processing files house utility functions whereas the Jupyter notebooks provide some usecases based on the dataset
described in the paper: 
__Recognition of breathing activity and medication
adherence using LSTM Neural Networks - BIBE 2019__

- The environment for this project can be replicated with the environment.yml file provided.

- The dataset used for this paper is only accessible by request, but the model works with any .wav file containing respiratory sounds.

# 
_Feel free to fork and use it with your dataset._



