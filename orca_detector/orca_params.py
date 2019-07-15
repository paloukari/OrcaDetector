"""
Global parameters for the OrcaVGGish model.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

from enum import IntEnum

# Click text

EPILOGUE = 'by Spyros Garyfallos, Ram Iyer, Mike Winton'


class DatasetType(IntEnum):
    """ Enumeration with the possible dataset types. """
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


# Paths to key volumes mapped into the Docker container
DATA_PATH = '/data/'
LIVE_FEED_PATH = '/data/live_feed/'
POSITIVE_SAMPLES_PATH = '/data/positive_samples/'
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

LOSS = 'categorical_crossentropy'
EPOCHS = 20
BATCH_SIZE = 64
MODEL_NAMES = ['vggish', 'logreg']
DEFAULT_MODEL_NAME = 'vggish'

# Model training - VGGish params
OPTIMIZER = 'adam'
LEARNING_RATE = 0.001  # SGD default LR=0.01; Adam default LR=0.001
DROPOUT = 0.4
FINAL_DENSE_NODES = 256
L2_REG_RATE = 0.01  # used for all Dense and Conv2D layers

# Model training - LogReg params
LOGREG_OPTIMIZER = 'adam'
LOGREG_LEARNING_RATE = 0.005

# Mel spectrogram hyperparameters
FILE_MAX_SIZE_SECONDS = 10.00
FILE_SAMPLING_SIZE_SECONDS = 0.98
DROPOUT = 0.4

# Live feed listener
LIVE_FEED_SEGMENT_SECONDS = 1
LIVE_FEED_SLEEP_SECONDS = 0
LIVE_FEED_ITERATION_SECONDS = 10

# TODO: Verify this is fixed to this value
LIVE_FEED_SAMPLING_RATE = 48000

# Dictionary of stream base URLs; used in building stream links
ORCASOUND_STREAMS = {
    'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab',
    'BushPoint': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point',
    'PortTownsend': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_port_townsend'
}

ORCASOUND_STREAMS_NAMES = list(ORCASOUND_STREAMS.keys()) + ['All']