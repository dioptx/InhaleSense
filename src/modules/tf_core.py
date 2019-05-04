from tensorflow import python as tf
from src.modules.io_tools import fetch_dataset
from src.modules.visualizer import do_heatmap
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import LSTM

annotations, dataset = fetch_dataset()


t_dataset = dataset[['Label', 'Sample']]



len_E = []
num_E = 0
len_I = []
num_I = 0

len_D = []
num_D = 0



image_dataset = []
label_dataset = []

for idx, row in dataset.iterrows():
    if row['Label'] == 'Noise':
        continue
    elif row['Label'] == 'Exhale':
        len_E.append(len(row['Sample'][0]))
        num_E += 1
    elif row['Label'] == 'Inhale':
        len_I.append(len(row['Sample'][0]))
        num_I += 1

    elif row['Label'] == 'Drug':
        len_D.append(len(row['Sample'][0]))
        num_D += 1



    label = row['Label']
    if label == 'Inhale':
        label = [1,0,0]
    elif label == 'Exhale':
        label = [0,1,0]
    elif label == 'Drug':
        label = [0,0,1]

    spect = row['SPECT'][0]

    image_dataset.append(spect)
    label_dataset.append([label for i in range(0, spect.shape[1])])


    # label = row['Label']
    # cepst = row['CEPST'][0]
    # do_heatmap(cepst, title='CEPST'+label )
    #
    # spect = row['SPECT'][0]
    # do_heatmap(spect, title='SPECT' + label)
    #
    # mfcc = row['MFCC'][0]
    # do_heatmap(mfcc, title='MFCC' + label)
    #
    # cwt = row['CWT'][0]
    # do_heatmap(cwt, title='cwt' + label)
    # wav = row['Sample']
    # do_heatmap(wav)


    print(row['Label'])


#omorfo prama 0
flat_list = []

for sublist in image_dataset:
    for item in sublist.T:
        flat_list.append(item)

#omorfo prama 1
flat_labels = []
for sublist in label_dataset:
    for item in sublist:
        flat_labels.append(item)

shift_step = 15


# for idx, feature in enumerate(flat_list):
#     window = flat_list[idx: idx + 15]
#     if len(window) == 15:
#         print(idx)
#     else:
#         print('Edge Boi')



def read_dataset(dataset, labels, test):
    if test:
        for idx, feature in enumerate(dataset[10001:20000]):
            window = dataset[idx: idx + shift_step]
            if len(window) == shift_step:
                yield np.array(window), np.array(labels[idx])
    else:
        for idx, feature in enumerate(dataset[0:10000]):
            window = dataset[idx: idx+ shift_step]
            if len(window) == shift_step:
                yield np.array(window), np.array(labels[idx])

def get_dataset(test):
    generator = lambda: read_dataset(flat_list, flat_labels, test)
    return tf.data.Dataset.from_generator(
        generator, (tf.float32, tf.int32), ((shift_step, 42), ( 3,))).batch(25)




# Create a model
model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(shift_step, 42)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

model.fit(get_dataset(test=False), epochs=1)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(get_dataset(test=True))
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print('\n# Generate predictions for 3 samples')
#predictions = model.predict(x_test[:3])
#print('predictions shape:', predictions.shape)

print('Exhale ',min(len_E), max(len_E), 'Num: ', num_E)
print('Inhale ',min(len_I), max(len_I), 'Num: ', num_I)
print('Drug ',min(len_D), max(len_D), 'Num: ', num_D)






