"""
Global parameters for the OrcaVGGish model.
"""

# Classification params
CLASSES = ['KillerWhale', 'FrasersDolphin']

# Number of classes needs to account for "Other"
NUM_CLASSES = len(CLASSES) + 1

# Weighting to account for imbalance when calculating loss
CLASS_WEIGHTS = {'KillerWhale': 1.,
                 'FrasersDolphin': 1.}

