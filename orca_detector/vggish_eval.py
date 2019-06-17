"""
Audio classification demo using embeddings from the pretrained VGGish model.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from https://github.com/DTaoo/VGGish
"""

import linecache
import numpy as np
import os
import sys

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from scipy.io import wavfile
from sklearn import svm

# project-specific imports
from vggish import VGGish
import vggish_preprocess

SOUND_FILE_PATH = '/data/'
TRAIN_FILE = os.path.join(SOUND_FILE_PATH, 'train.txt')
TEST_FILE  = os.path.join(SOUND_FILE_PATH, 'test.txt')

# for consistency and replication
np.random.seed(251)    

def load_embeddings(file, sound_extractor):
    """
    Use provided sound_extractor to generate embeddings for audio samples listed
    in a given file.
    """
    
    # TODO: Update to crawl our directory structure for wav files
    lines = linecache.getlines(file)
    sample_num = len(lines)
    seg_num = 60
    seg_len = 5  # 5 seconds
    data = np.zeros((seg_num * sample_num, 496, 64, 1))
    label = np.zeros((seg_num * sample_num,))

    for i in range(len(lines)):
        sound_file = os.path.join(SOUND_FILE_PATH, lines[i][:-7])
        sr, wav_data = wavfile.read(sound_file)

        length = sr * seg_len  # 5s segment
        range_high = len(wav_data) - length
        random_start = np.random.randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i * seg_num + j, :, :, :] = cur_spectro
            # TODO: get label from directory structure
            label[i * seg_num + j] = lines[i][-2]

    data = sound_extractor.predict(data)

    return data, label


if __name__ == '__main__':

    sound_extractor = VGGish(load_weights=True, 
                             weights='audioset',
                             include_top=False, 
                             pooling='avg').get_model()

    # load embeddings for training data
    print ("loading embeddings for training data...")
    training_data, training_label = load_embeddings(TRAIN_FILE, sound_extractor)

    # load embeddings for testing data
    print ("loading embeddings for testing data...")
    test_count = len(linecache.getlines(TEST_FILE))
    testing_data, testing_label = load_embeddings(TEST_FILE, sound_extractor)

    # perform Linear Support Vector Classification with the audio embeddings
    # obtained from the VGGish model (as a demonstration of how to use them).
    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(training_data, training_label.ravel())
    p_vals = clf.decision_function(testing_data)

    # calculate accuracy of the SVC model
    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:6000:60]
    p_vals = np.asarray(p_vals)

    for ii in range(test_count):
        scores = np.mean(p_vals[ii * 60:(ii + 1) * 60, :], axis=0)
        ind = np.argmax(scores)
        pred_labels[ii] = ind
    scores = gt == pred_labels
    score = np.mean(scores)
    print ("accuracy: {}".format(score))



