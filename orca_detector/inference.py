# -*- coding: future_fstrings -*-

"""
Main file to run inference with a pretrained model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import argparse
import datetime
import numpy as np
import orca_params
import os
import pickle
import tensorflow as tf

from database_parser import encode_labels, load_features
from keras.models import Sequential
from orca_params import DatasetType
from orca_utils import calculate_accuracies
from vggish_model import OrcaVGGish

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_network(num_classes, weights_path):
    """ 
    Instantiate trained model from given weights.
    Create the output shape based on num_classes.
    """

    sound_extractor = OrcaVGGish(load_weights=True,
                                 weights=weights_path,
                                 out_dim=num_classes,
                                 pooling='avg').get_model()

    return sound_extractor


def run(weights_path=None, predict_only=False, **params):

    print(f'TensorFlow version: {tf.VERSION}')
    print(f'Keras version: {tf.keras.__version__}')

    # load the dataset features and labels from disk.
    if not predict_only:
        test_features, test_labels = load_features(
            orca_params.DATA_PATH, DatasetType.TEST)
    else:
        # TODO: handle cases where we have features but no labels
        pass

    # load trained LabelEncoder and encode test set labels
    encoder_path = os.path.join(orca_params.DATA_PATH, 'label_encoder.p')
    if os.path.isfile(encoder_path):
        with open(encoder_path, 'rb') as f:
            print(f'Loading trained LabelEncoder from {encoder_path}')
            encoder = pickle.load(f)
    test_labels = encode_labels(test_labels, encoder)
    num_classes = len(encoder.classes_)
    
    # instantiate model and load weights
    if weights_path is None:
        weights_path = os.path.join(orca_params.OUTPUT_PATH,
                                    'orca_weights_latest.hdf5')
    model = create_network(num_classes, weights_path)

    results = model.predict(x=test_features,
                            batch_size=orca_params.BATCH_SIZE,
                            verbose=1)
    
    print(f'Labels predicted for {len(test_features)} samples')

    if not predict_only:
        # calculate classification metrics 
        pred_labels = np.argmax(results, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        calculate_accuracies(pred_labels,
                             true_labels,
                             run_timestamp=RUN_TIMESTAMP)
    else:
        pred_labels = np.argmax(results, axis=1)
        # TODO: do something with predictions on unlabeled audio


if __name__ == '__main__':
    
    # parse command line parameters and flags
    parser = argparse.ArgumentParser(description='OrcaDetector - W251 (Summer 2019)',
                        epilog='by Spyros Garyfallos, Ram Iyer, Mike Winton')
    parser.add_argument('--weights',
                        type=str,
                        help='Specify the weights path to use.')
    parser.add_argument('--predict_only', action='store_true',
                        help = 'Run inference for unlabeled audio.')
    args = parser.parse_args()
    
    if not args.predict_only:
        predict_only = False
    else:
        predict_only = True
    
    if args.weights:
        run(weights=arg.weights, predict_only=predict_only)
    else:
        run(predict_only=predict_only)
