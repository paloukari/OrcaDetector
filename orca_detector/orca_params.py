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
OUTPUT_PATH = '/results/'
WEIGHTS_PATH = '/vggish_weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = '/vggish_weights/vggish_audioset_weights.h5'

# Classification params
OTHER_CLASS = 'Other'
OTHER_CLASSES = []
REMOVE_CLASSES = ['BeardedSeal',  # (NEED TO UNDERSTAND)
                  'BlueWhale',  # 3 training samples
                  'CommersonsDolphin',  # 1 training sample
                  'FinlessPorpoise',  # 2 training samples
                  'GraySeal',  # 7 training samples
                  'HarbourSeal',  # 1 training sample
                  'HeavisidesDolphin',  # 14 training samples
                  'HoodedSeal',  # 2 training samples
                  'IrawaddyDolphin',  # 5 training samples
                  'JuanFernandezFurSeal',  # 4 training samples
                  'LeopardSeal',  # (NEED TO UNDERSTAND)
                  'MinkeWhale',  # 24 training samples
                  'NewZealandFurSeal',  # 2 training samples
                  'RibbonSeal',  # 45 training samples
                  'RingedSeal',  # 46 training samples
                  'SeaOtter',  # 2 training samples
                  'Short_FinnedPacificPilotWhale',  # (NEED TO UNDERSTAND)
                  'SpottedSeal',  # 22 training samples
                  'StellerSeaLion',  # 6 training samples
                  'TucuxiDolphin'  # 12 training samples
                 ]

# Weighting to account for imbalance when calculating loss
OPTIMIZER = 'sgd'
LEARNING_RATE = 0.001  # SGD default LR=0.01
LOSS = 'categorical_crossentropy'

# Model training
EPOCHS = 30
BATCH_SIZE = 64

# Model hyperparameters
FILE_MAX_SIZE_SECONDS = 10.00
FILE_SAMPLING_SIZE_SECONDS = 0.98
DROPOUT = 0.
