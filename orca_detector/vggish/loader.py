import tensorflow as tf
import random
import multiprocessing
from collections import defaultdict
import os
import numpy as np
import soundfile as sf
import pickle
from sklearn.preprocessing import LabelEncoder

quantum_size = 0.96


def quantize_sample(label, file):
    with sf.SoundFile(file) as wav_file:
        length = wav_file.frames / wav_file.samplerate
        file_parts = np.arange(0.0, length, quantum_size)
        return [[label, f"{file}:{segment:.2f}"] for segment in file_parts]


def quantize_samples(samples):
    quantized_samples = [quantize_sample(label, file) for [label, file] in samples]
    flat_quantized_samples = [item for sublist in quantized_samples for item in sublist]
    return flat_quantized_samples

def onehot(labels):
    classes = len(set(labels))
    encoder = LabelEncoder()
    encoder.fit(list(labels))
    encoded_labels = encoder.transform(labels)

    onehot_encoded_labels = np.zeros((len(encoded_labels), classes))
    onehot_encoded_labels[np.arange(len(encoded_labels)), encoded_labels] = 1

    return onehot_encoded_labels

def index_files(folder, split_percentage=.90):
    
    if os.path.exists('train.tmp') and os.path.exists('validate.tmp'):
        with open('train.tmp', 'rb') as fp:
            train_flattened = pickle.load(fp)
        with open ('validate.tmp', 'rb') as fp:
            validate_flattened = pickle.load(fp)

    else:

        all_samples = defaultdict(list)
        for (dirpath, dirnames, filenames) in os.walk(folder):
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

        validate_flattened = quantize_samples(validate_flattened)
        train_flattened = quantize_samples(train_flattened)
    
        with open('train.tmp', 'wb') as fp:
            pickle.dump(train_flattened, fp)

        with open('validate.tmp', 'wb') as fp:
            pickle.dump(validate_flattened, fp)
    
    train_labels, train_files = zip(*train_flattened)
    validate_labels, validate_files = zip(*validate_flattened)

    return train_files, onehot(train_labels), validate_files, onehot(validate_labels)

def _load_and_extract_features(audio_segments, labels):
    with tf.Session():
        file = audio_segments.eval()
    pass


def loader(audio_segment_list, label_list, batch_size=32, epoch=None):

    audio_segments = tf.convert_to_tensor(audio_segment_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int8)

    # Create dataset out of the 2 lists:
    data = tf.data.Dataset.from_tensor_slices((audio_segments, labels))

    data = data.shuffle(buffer_size=len(audio_segment_list))

    # Parse images and label
    data = data.map(_load_and_extract_features,
                    num_parallel_calls=multiprocessing.cpu_count()).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Batch, epoch, shuffle the data
    data = data.batch(batch_size)
    data = data.repeat(epoch)

    # Create iterator
    iterator = data.make_one_shot_iterator()

    #init_op = iterator.make_initializer()

    # Next element Op
    next_element = iterator.get_next()
    return next_element  # , init_op


def test():
    train_files, train_labels, validate_files, validate_labels = index_files('../../data')

    train_loader = loader(train_files, train_labels)
    train_loader.get_next()


test()
