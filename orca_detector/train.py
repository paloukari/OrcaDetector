"""
Main file to train a model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import h5py
import os
import tensorflow as tf
import datetime
from keras.models import Sequential

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# project-specific imports
import database_parser
import orca_params
from database_parser import load_dataset
from generator import WavDataGenerator
from orca_utils import plot_train_metrics, save_model
from vggish_model import OrcaVGGish

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')

def print_framework_versions():
    print('TensorFlow version: {}'.format(tf.VERSION))
    print('Keras version: {}'.format(tf.keras.__version__))

def create_network():
    """ Instantiate but don't yet fit the model."""

    sound_extractor = OrcaVGGish(load_weights=True, 
                                 weights='audioset',
                                 pooling='avg').get_model()

    return sound_extractor

def run(**params):

    print_framework_versions()
    
    # load the dataset mappings from disk.
    train_files, train_labels = load_dataset(orca_params.DATA_PATH, 'train')
    validate_files, validate_labels = load_dataset(orca_params.DATA_PATH, 'validate')
    
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
    print('Saved loss plot -> {}'.format(loss_fig_path))
    print('Saved accuracy plot -> {}'.format(acc_fig_path))

    # save json model config file and trained weights to disk
    json_path, weights_path = save_model(model, RUN_TIMESTAMP)
    print('Saved json config -> {}'.format(json_path))
    print('Saved weights -> {}'.format(weights_path))


if __name__ == '__main__':
    run()
