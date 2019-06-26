"""
Global parameters for the OrcaVGGish model.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

from enum import IntEnum


class DatasetType(IntEnum):
    """ Enumeration with the possible dataset types. """
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


# Paths to key volumes mapped into the Docker container
DATA_PATH = '/data/'
HDF5_FILENAME = 'features_dataset.hdf5'
OUTPUT_PATH = '/results/'
WEIGHTS_PATH = '/vggish_weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = '/vggish_weights/vggish_audioset_weights.h5'

# Classification params
CLASSES = ['KillerWhale', 'FrasersDolphin']

# Number of classes needs to account for "Other"
NUM_CLASSES = len(CLASSES) + 1

# Weighting to account for imbalance when calculating loss
# TODO: update weights based on observed balance
CLASS_WEIGHTS = {0: 1., 1: 1., 2: 1.}
OPTIMIZER = 'sgd'
LOSS = 'categorical_crossentropy'

# Model training
EPOCHS = 5
BATCH_SIZE = 32

# IMPORTANT: it's not straightforward to reduce this below 5, due to dependencies
# between params.NUM_FRAMES and params.EXAMPLE_WINDOW_SECONDS (I think).  It will
# take some debugging to work with shorter samples, but it may be worthwhile since
# our dataset has quite a few samples < 5 sec.
FILE_MAX_SIZE_SECONDS = 10.00
FILE_SAMPLING_SIZE_SECONDS = 0.98
