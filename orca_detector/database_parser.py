# -*- coding: future_fstrings -*-

"""
File to parse and label datafiles for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import numpy as np
import os
import pickle
import pprint
import random
import re
import soundfile as sf

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# project-specific imports
import orca_params
from orca_params import DatasetType


def _label_files(data_path=orca_params.DATA_PATH):
    """
        Walks the data_path looking for *.wav files and builds a dictionary of files and their
        respective labels based on subdirectory names.

        ARGS:
            data_path = directory root to walk

        RETURNS:
            dictionary with key=label name; value=list of associated files
    """

    # build a defaultdict of all of the samples read from disk.
    # key will be the class label (text). Value will be a list of all file paths
    total_files = 0
    all_samples = defaultdict(list)
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        # extract audio filenames
        filenames = [filename for filename in filenames if os.path.splitext(filename)[
            1].lower() == '.wav']
        total_files += len(filenames)
        if len(filenames) == 0:
            continue  # try next folder

        # extract folder names as labels from a path that looks like:
        #   /data/MarineMammalName/1975
        path, year_folder = os.path.split(dirpath)
        _, label = os.path.split(path)
        # strip non alphanumeric characters
        label = re.sub('\W+', '', label)

        all_samples[label].extend(
            [os.path.join(dirpath, file) for file in filenames])
        # print('Loaded data from {}, mapped to label={}'.format(dirpath, label))

    print('In walking directory, observed {} labels for {} audio files.'.format(
        len(all_samples), total_files))
    return all_samples


def _backup_datafile(file_path, suffix='-old'):
    """
        Rename file_path to file_path+suffix to provide one level of "undo".

        ARGS:
            file_path = path of file to be renamed
            suffix = string to append during renaming

        RETURNS:
            nothing
    """

    if os.path.exists(file_path):
        renamed_file = '{}{}'.format(file_path, suffix)
        os.rename(file_path, renamed_file)
        print('Renamed {} to {}'.format(file_path, renamed_file))


def _save_indices(flattened_file, data_path=orca_params.DATA_PATH, dataset_type=None, backup=True):
    """
        Saves file of indices for a specific dataset (train/val/test) to disk.

        ARGS:
            flattened_file = path of file to be saved
            dataset_type = one of DatasetType enum
            backup = boolean whether to backup (vs. overwrite)

        RETURNS:
            nothing
    """

    # validate dataset_type is valid
    if dataset_type not in [item.value for item in DatasetType]:
        raise ValueError('ERROR: invalid DatasetType specified.')
    if dataset_type == DatasetType.TRAIN:
        indices_file = os.path.join(data_path, 'train_map.p')
    elif dataset_type == DatasetType.VALIDATE:
        indices_file = os.path.join(data_path, 'validate_map.p')
    elif dataset_type == DatasetType.TEST:
        indices_file = os.path.join(data_path, 'test_map.p')

    # rename old files instead of overwriting.
    if backup:
        _backup_datafile(indices_file)

    with open(indices_file, 'wb') as fp:
        pickle.dump(flattened_file, fp)
        print('Saved dataset indices (metadata) to {}'.format(indices_file))


def _quantize_sample(label, file, sample_len=orca_params.FILE_MAX_SIZE_SECONDS):
    """
        Splits up a given file into non-overlapping segments of the specified length.
        Returns a list containing (label, 'file:start:frames') of each segment.

        Final trailing segments that are too short are dropped.

        ARGS:
            label = string name
            file = *.wav audio file
            sample_length = length of audio segments to be identified

        RETURNS:
            list of audio segments
    """

    with sf.SoundFile(file) as wav_file:
        # make sure sample is long enough
        min_frames = int(sample_len * wav_file.samplerate)  # e.g. 2 * 16000
        if wav_file.frames > min_frames:
            file_parts = np.arange(0, wav_file.frames, min_frames)
            sample_list = [[label, '{}:{}:{}'.format(
                file, int(start), min_frames)] for start in file_parts]

            # truncate final sample which will be shorter than min required for a spectrogram
            del sample_list[-1]
            # add it back in with some overlap
            sample_list.append([label,'{}:{}:{}'.format(
                file, int(wav_file.frames - min_frames), min_frames)])
            return sample_list
        else:
            return []


def _quantize_samples(samples, sample_len=orca_params.FILE_MAX_SIZE_SECONDS):
    """
        Quantizes a list of audio files into short segments

        ARGS:
            samples = list of (label, file path)

        RETURNS:
            flattened list
    """
    quantized_samples = [_quantize_sample(label, file) for [
        label, file] in samples]
    flat_quantized_samples = [
        item for sublist in quantized_samples for item in sublist]
    return flat_quantized_samples


def _flatten_and_quantize_dataset(dataset):
    """
        Flattens and quantizes audio segments from each file in the dataset (train/val/test)

        ARGS:
            dataset = list of (label, file path)

        RETURNS:
            flattened list of audio segments from the specified files
    """

    # Create lists with each element looking like:
    #   ['SpermWhale', '/data/SpermWhale/1985/8500901B.wav']
    dataset_flattened = [[label, file]
                         for label in dataset.keys() for file in dataset[label]]
    dataset_quantized = _quantize_samples(dataset_flattened)
    print('\nQuantized {} audio segments from {} sample files.'
          .format(len(dataset_quantized), len(dataset_flattened)))

    return dataset_quantized


def _onehot(labels, desired_classes=orca_params.CLASSES, data_path=orca_params.DATA_PATH):
    """
        One-hot encodes labels in preparation for passing to a Keras model.

        All classes that aren't in orca_params.CLASSES are mapped to 'Other'.

        ARGS:
            labels = list of all observed label names (may be a tuple)
            data_path = path to directory where the indices files are found

        RETURNS:
            np.array[observations, classes] representing the labels
    """

    # TODO: rework the OHE logic so that it doesn't happen at runtime in the generator.

    # convert to lists (zip generates tuples)
    labels = list(labels)

    # with label encoder, map all undesired classes to "Other"
    ohe_classes = desired_classes + ['Other']
    encoder = LabelEncoder()
    encoder.fit(ohe_classes)

    # create a list holding the int class labels
    for i in range(len(labels)):
        if not labels[i] in orca_params.CLASSES:
            labels[i] = 'Other'
    encoded_labels = encoder.transform(labels)

    # build into a numpy array to return
    onehot_encoded_labels = np.zeros((len(encoded_labels), len(ohe_classes)))
    onehot_encoded_labels[np.arange(len(encoded_labels)), encoded_labels] = 1

    # save LabelEncoder so inverse transforms can be recovered
    label_encoder_file = os.path.join(data_path, 'label_encoder.p')
    # rename old files instead of overwriting.
    _backup_datafile(label_encoder_file)
    with open(label_encoder_file, 'wb') as fp:
        pickle.dump(encoder, fp)
        print('Saved label encoder to {}'.format(label_encoder_file))

    return onehot_encoded_labels


def train_val_test_split(data_path=orca_params.DATA_PATH,
                         train_percentage=.70,
                         validate_percentage=0.20):
    """
        Index files and create a train/val/test split.  Note that label one-hot
        encoding is *not* done at this point, nor are undesired classes converted
        to "Other".  That is done when loading the dataset.
    """

    # TODO: change train/val/test split to happen *after* quantization

    all_samples = _label_files(data_path=orca_params.DATA_PATH)

    datasets = {DatasetType.TRAIN: defaultdict(list),
                DatasetType.VALIDATE: defaultdict(list),
                DatasetType.TEST: defaultdict(list)}

    # do a stratified train/val/test split
    for label, files in all_samples.items():
        if len(files) < 10:
            continue  # don't bother to shuffle
        random.shuffle(files)
        num_train_files = int((len(files) + 1) * train_percentage)
        num_validate_files = int((len(files) + 1) * validate_percentage)
        datasets[DatasetType.TRAIN][label] = \
            files[: num_train_files]
        datasets[DatasetType.VALIDATE][label] = \
            files[num_train_files: num_train_files + num_validate_files]
        datasets[DatasetType.TEST][label] = \
            files[num_train_files + num_validate_files:]

#     for key, val in datasets.items():
#         for l in val:
#             print('DEBUG: key={}, label={}, num entries={}'.format(key.name, l, len(datasets[key][l])))

    # quantize and flatten each dataset
    for dataset_type, contents in datasets.items():
        flattened_dataset = _flatten_and_quantize_dataset(contents)
        _save_indices(flattened_dataset, data_path, dataset_type, backup=True)


def load_dataset(data_path=orca_params.DATA_PATH, dataset_type=None):
    """
        Load the lists of files and labels for train, val, or test set.  One-hot
        encoding of the labels is performed at this time based on the desired
        classes specified in orca_params.CLASSES (all others are mapped to 'Other').
        The pickled LabelEncoder() is also saved so that an inverse transform can
        later be performed with it if desired.
    """

    if dataset_type not in DatasetType:
        raise ValueError('ERROR: invalid DatasetType specified.')
    if dataset_type == DatasetType.TRAIN:
        indices_file = os.path.join(data_path, 'train_map.p')
    elif dataset_type == DatasetType.VALIDATE:
        indices_file = os.path.join(data_path, 'validate_map.p')
    elif dataset_type == DatasetType.TEST:
        indices_file = os.path.join(data_path, 'test_map.p')

    if os.path.exists(indices_file):
        with open(indices_file, 'rb') as f:
            flattened = pickle.load(f)
        print('\nLoaded {} dataset from {}'.format(
            dataset_type.name, indices_file))
    else:
        raise Exception(
            'ERROR: run database_parser.py to generate datafiles first.')

    labels, files = zip(*flattened)

    return files, _onehot(labels)


if __name__ == '__main__':

    # generate and save index files
    train_val_test_split()

    # load all datasets to validate
    for dataset_type in DatasetType:
        files, labels = load_dataset(dataset_type=dataset_type)
        print('\nDatasetType={}'.format(dataset_type.name))
        print('Sample files:')
        pprint.pprint(files[:5])
        print('Counts by label:')
        print(np.sum(labels, axis=0).astype(int))
    