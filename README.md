# InhaleSense

The goal of this project is to produce a deep learning model that discovers and classifies sections of interest in audio files.


The dataset used for the initial stage of the implementation is not provided. 

[Research Gate & Publication Link](https://www.researchgate.net/publication/335135907_Recognition_of_breathing_activity_and_medication_adherence_using_LSTM_Neural_Networks)

### Authors of the publication:
- Dionisis Pettas (dennis.petta@gmail.com)
- Stavros Nousias (nousias.stavros@gmail.com)
- Evangelia Zacharaki (ezachar@upatras.gr)

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



### Demo with dataset included

An extended version of the code also including and comparing other methods is made available:

- https://codeocean.com/capsule/8383844/tree/ 
- https://github.com/snousias/Revisiting-Content-Based-Audio-Classification-for-Asthma-Medication-Adherence/tree/master


### Dataset on IEEE Dataport

The dataset is available in IEEE Dataport and the dataset format can be summarized as follows: 

Link:

- https://ieee-dataport.org/documents/respiratory-and-drug-actuation-dataset

Generic format:
```
Filename, Class, Sample index at the beginning of the acoustic event, Sample index at the end of the acoustic event

```

Example:

```
rec2018-01-22_17h41m33.475s.wav,Exhale,6015,17437
rec2018-01-22_17h41m33.475s.wav,Inhale,20840,31655
rec2018-01-22_17h41m33.475s.wav,Drug,31898,37610
rec2018-01-22_17h41m33.475s.wav,Exhale,43686,59969
rec2018-01-22_17h41m49.809s.wav,Inhale,5043,17316
rec2018-01-22_17h41m49.809s.wav,Drug,18288,24364
rec2018-01-22_17h41m49.809s.wav,Exhale,31412,46724
rec2018-01-22_17h42m07.718s.wav,Exhale,303,9782
rec2018-01-22_17h42m07.718s.wav,Inhale,16951,28010
```

# 
_Feel free to fork and use it with your dataset._
