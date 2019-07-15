# -*- coding: future_fstrings -*-

"""
Main file to run inference with a pretrained model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import click
import datetime
import numpy as np
import orca_params
import os
import pickle
import tensorflow as tf

from database_parser import encode_labels, load_features
from keras.models import Sequential
from logreg_model import OrcaLogReg
from orca_params import DatasetType
from orca_utils import calculate_accuracies
from vggish_model import OrcaVGGish

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_network(model_name, encoder_path, weights_path):
    """ 
    Instantiate trained model from given weights and label encoder.
    Create the output shape based on num_classes of the saved encoder.
    """
    
    # load trained LabelEncoder and encode test set labels
    if os.path.isfile(encoder_path):
        with open(encoder_path, 'rb') as f:
            print(f'Loading trained LabelEncoder from {encoder_path}')
            encoder = pickle.load(f)
    num_classes = len(encoder.classes_)

    if model_name == 'vggish':
        model = OrcaVGGish(load_weights=True,
                           weights=weights_path,
                           out_dim=num_classes,
                           pooling='avg').get_model()
    elif model_name == 'logreg':
        model = OrcaLogReg(load_weights=True,
                           weights=weights_path,
                           out_dim=num_classes).get_model()
    else:
        raise Exception('No model specified.  Use `--model-name` arg.')

    return model, encoder

@click.command(help="Performs inference.",
               epilog=orca_params.EPILOGUE)
@click.option('--model-name',
              help='Specify the model name to use.',
              default=orca_params.DEFAULT_MODEL_NAME,
              show_default=True,
              type=click.Choice(
                  choices=orca_params.MODEL_NAMES))
@click.option('--label-encoder-path',
              help='Specify the label encoder path to use.', 
              default=os.path.join(orca_params.OUTPUT_PATH,
                                    'label_encoder_latest.p'), 
              show_default=True)
@click.option('--weights-path',
              help='Specify the weights path to use.', 
              default=os.path.join(orca_params.OUTPUT_PATH,
                                    'orca_weights_latest.hdf5'), 
              show_default=True)
@click.option('--predict-only',
              help='Run inference for unlabeled audio.',
              show_default=True,
              is_flag=True,
              default=False)
def infer(model_name, label_encoder_path, weights_path, predict_only):

    print(f'TensorFlow version: {tf.VERSION}')
    print(f'Keras version: {tf.keras.__version__}')

    # load the dataset features and labels from disk.
    if not predict_only:
        test_features, test_labels = load_features(
            orca_params.DATA_PATH, DatasetType.TEST)
    else:
        # TODO: handle cases where we have features but no labels
        pass

    # instantiate model and load weights
    model, encoder = create_network(model_name,
                                    label_encoder_path,
                                    weights_path)

    test_labels = encode_labels(test_labels, encoder)

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
