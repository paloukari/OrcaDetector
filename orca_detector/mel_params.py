"""
Global parameters for generation of log mel spectrograms.
Adapted from  https://github.com/tensorflow/models/tree/master/research/audioset
"""

# Architectural constants.
NUM_FRAMES = 496  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 4.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 4.96     # with zero overlap.
