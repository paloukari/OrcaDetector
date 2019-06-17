"""
Global parameters for the OrcaVGGish model.
"""

# Classification params
CLASSES = ['KillerWhale', 'FrasersDolphin']

# Number of classes needs to account for "Other"
NUM_CLASSES = len(CLASSES) + 1

# Weighting to account for imbalance when calculating loss
#TODO: this will need to have integer keys after one-hot encoding
CLASS_WEIGHTS = {'KillerWhale': 1.,
                 'FrasersDolphin': 1.}

# Model training
EPOCHS = 10
BATCH_SIZE = 32
FILE_MAX_SIZE_SECONDS = 5
 
