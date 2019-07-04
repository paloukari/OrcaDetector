# -*- coding: future_fstrings -*-

"""
Main file to train a model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

from vggish_model import OrcaVGGish
from orca_utils import plot_train_metrics, save_model
from orca_params import DatasetType
from database_parser import load_features, create_label_encoding, encode_labels
import orca_params
import os
import tensorflow as tf
import datetime
from keras.models import Sequential

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_network(classes):
    """ 
    Instantiate but don't yet fit the model.
    Create the output shape based on the classes length
    """

    sound_extractor = OrcaVGGish(load_weights=True,
                                 weights='audioset',
                                 out_dim=len(classes),
                                 pooling='avg').get_model()

    return sound_extractor


def run(**params):

    print(f'TensorFlow version: {tf.VERSION}')
    print(f'Keras version: {tf.keras.__version__}')

    # load the dataset features from disk.
    train_features, train_labels = load_features(
        orca_params.DATA_PATH, DatasetType.TRAIN)
    validate_features, validate_labels = load_features(
        orca_params.DATA_PATH, DatasetType.VALIDATE)

    # one hot and filter labels
    classes = set(train_labels).union(set(validate_labels))
    # train and save encoder to disk for later use
    encoder = create_label_encoding(list(classes))
    train_labels = encode_labels(train_labels, encoder)
    validate_labels = encode_labels(validate_labels, encoder)

    model = create_network(classes)

    history = model.fit(x=train_features,
                        y=train_labels,
                        validation_data=(validate_features, validate_labels),
                        epochs=orca_params.EPOCHS,
                        verbose=1)

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
