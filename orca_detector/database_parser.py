import os
import pickle
import random
from collections import defaultdict

import numpy as np
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

# project-specific imports
import orca_params
import vggish_params as params

def _quantize_sample(label, file):
    with sf.SoundFile(file) as wav_file:
        # more than 1 sec
        if wav_file.frames > wav_file.samplerate:
            frames = int(params.FILE_MAX_SIZE_SECONDS * wav_file.samplerate)
            file_parts = np.arange(0, wav_file.frames, frames)
            return [[label, '{}:{}:{}'.format(file, int(start), frames)] for start in file_parts]
            #return [[label, f"{file}:{int(start)}:{frames}"] for start in file_parts]
        else:
            return []

def _quantize_samples(samples):
    quantized_samples = [_quantize_sample(label, file) for [
        label, file] in samples]
    flat_quantized_samples = [
        item for sublist in quantized_samples for item in sublist]
    return flat_quantized_samples

def _onehot(labels):
    # TODO:
    #  get rid of punctuation and spaces when converting directory names to class names
    #  then check list of desired labels from orca_params.CLASSES
    #  then one-hot encode all other classes as "Other"
    #
    #  Also, may need to convert this to a class and keep a member variable with the
    #  one hot encoding mapping to make sure that train and validate use the same
    #  mapping.
    #
    #  Also, when training, we need to write this mapping to disk as a text or json file
    #  so that it can be picked up and loaded when running a test set.
    
    classes = len(set(labels))
    encoder = LabelEncoder()
    encoder.fit(list(labels))
    encoded_labels = encoder.transform(labels)

    onehot_encoded_labels = np.zeros((len(encoded_labels), classes))
    onehot_encoded_labels[np.arange(len(encoded_labels)), encoded_labels] = 1

    return onehot_encoded_labels


def index_files(folder, split_percentage=.90):

    # TODO: figure out how to handle test set
    train_indices_file = os.path.join(folder, 'train.tmp')
    validate_indices_file = os.path.join(folder, 'validate.tmp')
    if os.path.exists(train_indices_file) and os.path.exists(validate_indices_file):
        with open(train_indices_file, 'rb') as fp:
            train_flattened = pickle.load(fp)
        with open(validate_indices_file, 'rb') as fp:
            validate_flattened = pickle.load(fp)

    else:

        all_samples = defaultdict(list)
        for (dirpath, dirnames, filenames) in os.walk(folder):
            filenames = [filename for filename in filenames if os.path.splitext(filename)[1].lower() == '.wav']
            
            if len(filenames) == 0:
                continue
            path, folder = os.path.split(dirpath)
            path, label = os.path.split(path)
            all_samples[label].extend(
                [os.path.join(dirpath, file) for file in filenames])

        train = defaultdict(list)
        validate = defaultdict(list)

        for label, files in all_samples.items():
            if len(files) < 10:
                continue
                random.shuffle(files)
            train_files = int((len(files)+1)*split_percentage)
            train[label] = files[:train_files]
            validate[label] = files[train_files:]

        train_flattened = [[label, file]
                           for label in train.keys() for file in train[label]]
        validate_flattened = [[label, file]
                              for label in train.keys() for file in validate[label]]

        validate_flattened = _quantize_samples(validate_flattened)
        train_flattened = _quantize_samples(train_flattened)

        with open(train_indices_file, 'wb') as fp:
            pickle.dump(train_flattened, fp)

        with open(validate_indices_file, 'wb') as fp:
            pickle.dump(validate_flattened, fp)

    train_labels, train_files = zip(*train_flattened)
    validate_labels, validate_files = zip(*validate_flattened)

    return train_files, _onehot(train_labels), validate_files, _onehot(validate_labels)
