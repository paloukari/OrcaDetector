# -*- coding: future_fstrings -*-

"""
Main file to train a model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

from vggish_model import OrcaVGGish
from orca_utils import plot_train_metrics, save_model
from orca_params import DatasetType
from generator import WavDataGenerator
from database_parser import load_dataset
import orca_params
import database_parser
import h5py
import os
import tensorflow as tf
import datetime
from keras.models import Sequential

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# project-specific imports

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_network():
    """ Instantiate but don't yet fit the model."""

    sound_extractor = OrcaVGGish(load_weights=True,
                                 weights='audioset',
                                 pooling='avg').get_model()

    return sound_extractor


def run(**params):

    print(f'TensorFlow version: {tf.VERSION}')
    print(f'Keras version: {tf.keras.__version__}')

    # load the dataset mappings from disk.
    train_files, train_labels = load_dataset(
        orca_params.DATA_PATH, DatasetType.TRAIN)
    validate_files, validate_labels = load_dataset(
        orca_params.DATA_PATH, DatasetType.VALIDATE)

    training_generator = WavDataGenerator(train_files,
                                          train_labels,
                                          shuffle=True,
                                          **params)

    # test the generator
    training_generator.__getitem__(0)

    validation_generator = WavDataGenerator(validate_files,
                                            validate_labels,
                                            shuffle=True,
                                            **params)

    model = create_network()

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  class_weight=orca_params.CLASS_WEIGHTS,
                                  epochs=orca_params.EPOCHS,
                                  use_multiprocessing=True,
                                  verbose=1,
                                  workers=1)

    # save loss and accuracy plots to disk
    loss_fig_path, acc_fig_path = plot_train_metrics(history, RUN_TIMESTAMP)
    print(f'Saved loss plot -> {loss_fig_path}')
    print(f'Saved accuracy plot -> {acc_fig_path}')

    # save json model config file and trained weights to disk
    json_path, weights_path = save_model(model, RUN_TIMESTAMP)
    print(f'Saved json config -> {json_path}')
    print(f'Saved weights -> {weights_path}')


if __name__ == '__main__':
    run()
