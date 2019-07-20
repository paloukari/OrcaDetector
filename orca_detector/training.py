# -*- coding: future_fstrings -*-

"""
Main file to train a model for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import click
import datetime
import orca_params
import os
import tensorflow as tf

from database_parser import load_features, create_label_encoding, encode_labels
from keras.models import Sequential
from logreg_model import OrcaLogReg
from orca_params import DatasetType
from orca_utils import plot_train_metrics, save_model_config, create_or_replace_symlink
from vggish_model import OrcaVGGish

from keras.callbacks import TensorBoard, ModelCheckpoint, \
    LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RUN_TIMESTAMP = datetime.datetime.now().isoformat('-')


def create_network(model_name, classes):
    """ 
    Instantiate but don't yet fit the model.
    Create the output shape based on the classes length
    """

    if model_name == 'vggish':
        model = OrcaVGGish(load_weights=True,
                           weights='audioset',
                           out_dim=len(classes),
                           pooling='avg').get_model()
    elif model_name == 'logreg':
        model = OrcaLogReg(load_weights=False,
                           out_dim=len(classes)).get_model()
    else:
        raise Exception('No model specified.  Use `--model_name` arg.')
        
    return model

@click.command(help="Trains the Orca Detector model.",
               epilog=orca_params.EPILOGUE)
@click.option('--model-name',
              help='Specify the model name to use.',
              default=orca_params.DEFAULT_MODEL_NAME,
              show_default=True,
              type=click.Choice(orca_params.MODEL_NAMES))
def train(model_name):

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
    encoder = create_label_encoding(list(classes), run_timestamp=RUN_TIMESTAMP)
    train_labels = encode_labels(train_labels, encoder)
    validate_labels = encode_labels(validate_labels, encoder)
    print(f'\nTraining to detect the following classes:\n{encoder.classes_}')
    print(f'Removed the following classes:\n{orca_params.REMOVE_CLASSES}')
    print(f'Mapped the following classes to "{orca_params.OTHER_CLASS}":\n' \
          f'{orca_params.OTHER_CLASSES}\n')

    model = create_network(model_name, classes)

    results_folder = os.path.join(orca_params.OUTPUT_PATH, model_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_folder = os.path.join(results_folder, RUN_TIMESTAMP)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    weight_path = os.path.join(output_folder, orca_params.BEST_WEIGHT_FILE_NAME)

    
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=orca_params.EARLY_STOPPING_PATIENCE)

    tensorboard = TensorBoard(log_dir=os.path.join(
        orca_params.OUTPUT_PATH, orca_params.TENSORBOARD_BASE_FOLDER, model_name, RUN_TIMESTAMP))

    dynamicLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=orca_params.LEARNING_RATE/100)

    callbacks_list = [tensorboard, checkpoint, dynamicLR, early]

    history = model.fit(x=train_features,
                        y=train_labels,
                        validation_data=(validate_features, validate_labels),
                        epochs=orca_params.EPOCHS,
                        batch_size=orca_params.BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1)

    # save loss and accuracy plots to disk
    loss_fig_path, acc_fig_path = plot_train_metrics(history, output_folder)
    print(f'Saved loss plot -> {loss_fig_path}')
    print(f'Saved accuracy plot -> {acc_fig_path}')

    # save json model config file
    json_path = save_model_config(model, history, output_folder)
    print(f'Saved json config -> {json_path}')

    # Create symbolic link to the most recent weights (to use for inference)
    symlink_path = os.path.join(orca_params.OUTPUT_PATH, model_name,
                                   'weights.best.hdf5')
    create_or_replace_symlink(weight_path, symlink_path)
    print(f'Created symbolic link to final weights as {symlink_path}')