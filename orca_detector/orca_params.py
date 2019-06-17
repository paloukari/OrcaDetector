"""
Global parameters for the OrcaVGGish model.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

# Classification params
CLASSES = ['KillerWhale', 'FrasersDolphin']

# Number of classes needs to account for "Other"
NUM_CLASSES = len(CLASSES) + 1

# Weighting to account for imbalance when calculating loss
#TODO: update weights based on observed balance
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
FILE_MAX_SIZE_SECONDS = 5
 
