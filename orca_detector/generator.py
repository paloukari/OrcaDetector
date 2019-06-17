"""
Keras custom generators for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import keras
import librosa
import numpy as np
import resampy
import soundfile as sf

# project-specific imports
import orca_params
import vggish_params as params
from mel_features import frame, log_mel_spectrogram

def _waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
          (multi-channel, with the outer dimension representing channels).
          Each sample is generally expected to lie in the range [-1.0, +1.0],
          although this is not required.
        sample_rate: Sample rate of data.

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is params.STFT_HOP_LENGTH_SECONDS.
    """
    
    # Convert to mono.
    if len(data.shape) > 1:
        print('DEBUG: before={}'.format(data.shape))
        data = np.mean(data, axis=1)
        print('DEBUG: after={}'.format(data.shape))
        
    # Resample to the rate assumed by VGGish.
    if sample_rate != params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(data,
                                  audio_sample_rate=params.SAMPLE_RATE,
                                  log_offset=params.LOG_OFFSET,
                                  window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                  hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                  num_mel_bins=params.NUM_MEL_BINS,
                                  lower_edge_hertz=params.MEL_MIN_HZ,
                                  upper_edge_hertz=params.MEL_MAX_HZ )

    # Frame features into examples.
    features_sample_rate  = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length    = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples      = frame(log_mel,
                                  window_length=example_window_length,
                                  hop_length=example_hop_length)
    
    return log_mel_examples


class WavDataGenerator(keras.utils.Sequence):
    """Generates data for Keras from Wav files"""

    def __init__(self, files, labels, shuffle=True):
        """Initialization"""
        
        self.files = files
        self.labels = labels  # one-hot encoded
        self.n_classes = labels.shape[1]
        self.n_batches = int(np.floor(len(self.files) / orca_params.BATCH_SIZE))
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        return self.n_batches

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        batch_indices = self.indices[index * orca_params.BATCH_SIZE: \
                                     (index + 1) * orca_params.BATCH_SIZE]

        # Find list of IDs
        files_temp  = [self.files[k] for k in batch_indices]
        labels_temp = [self.labels[k] for k in batch_indices]

        # Generate data
        X = self.__extract_features(files_temp)

        return X, labels_temp

    def on_epoch_end(self):
        """Updates indices after each epoch"""

        self.indices = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __extract_features(self, files_temp):
        """
            Generates data containing batch_size samples.
            
            Returns X : (n_samples, * (frames_size, bands_size))
        """

        X = []
        # Generate data from the appropriate segment of the audio file
        for i, file in enumerate(files_temp):
            data, sr = sf.read(file.split(':')[0], 
                               start = int(file.split(':')[1]), 
                               frames = int(file.split(':')[2]))
            # Transform to logmel format and store sample
            X.append(_waveform_to_examples(data, sr))

        return X
