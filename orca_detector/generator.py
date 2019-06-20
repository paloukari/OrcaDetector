"""
Keras custom generators for the Orca project.

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from https://github.com/DTaoo/VGGish
"""

import keras
import librosa
import numpy as np
import pprint
import resampy
import soundfile as sf

# project-specific imports
import mel_params
import orca_params
from mel_features import frame, log_mel_spectrogram

def _waveform_to_mel_spectrogram_segments(data, sample_rate):
    """
    Converts audio from a single wav file into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
          (multi-channel, with the outer dimension representing channels).
          Each sample is generally expected to lie in the range [-1.0, +1.0],
          although this is not required. Shape is (num_frame, )
        sample_rate: Sample rate of data.

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is mel_params.STFT_HOP_LENGTH_SECONDS.

    IMPORTANT: if data.shape < (80000, ) then log_mel_examples.shape=(0, 496, 64).
        The zero is problematic downstream, so code will have to check for that.
    """
    
    # Convert to mono if necessary.
    if len(data.shape) > 1:
        print('DEBUG: audio channels before={}'.format(data.shape))
        data = np.mean(data, axis=1)
        print('DEBUG: audio channels after={}'.format(data.shape))
        
    # Resample to the rate assumed by VGGish.
    if sample_rate != mel_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, mel_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(data,
                                  audio_sample_rate=mel_params.SAMPLE_RATE,
                                  log_offset=mel_params.LOG_OFFSET,
                                  window_length_secs=mel_params.STFT_WINDOW_LENGTH_SECONDS,
                                  hop_length_secs=mel_params.STFT_HOP_LENGTH_SECONDS,
                                  num_mel_bins=mel_params.NUM_MEL_BINS,
                                  lower_edge_hertz=mel_params.MEL_MIN_HZ,
                                  upper_edge_hertz=mel_params.MEL_MAX_HZ )

    # Frame features into examples.
    features_sample_rate  = 1.0 / mel_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(mel_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length    = int(round(mel_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    
    # If log_mel.shape[0] < mel_params.NUM_FRAMES, log_mel_examples will return
    #   an array with log_mel_examples.shape[0] = 0
    log_mel_examples = frame(log_mel,
                             window_length=example_window_length,
                             hop_length=example_hop_length)
    
    # print('DEBUG: data.shape={}'.format(data.shape))
    # print('DEBUG: log_mel_examples.shape={}'.format(log_mel_examples.shape))
    if log_mel_examples.shape[0] == 0:
        print('\nWARNING: audio sample too short! Using all zeros for that example.\n')
    return log_mel_examples


class WavDataGenerator(keras.utils.Sequence):
    """
        Generates data for Keras from wav files
    """

    def __init__(self, files, labels, shuffle=True):
        """Initialization"""
        
        self.files = files
        self.labels = labels  # one-hot encoded
        self.num_classes = labels.shape[1]
        self.batch_size = orca_params.BATCH_SIZE
        self.num_batches = int(np.floor(len(self.files) / self.batch_size))
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        return self.num_batches

    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Args:
            index = batch number being requested
        Returns:
            X = np.array of features (batch_size, num_frames, num_bands, 1)
            batch_labels = np.array of labels (batch_size, )
        """

        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size: \
                                     (index + 1) * self.batch_size]

        # Store labels in a numpy array (batch_size, num_classes)
        batch_labels = np.zeros((len(batch_indices), self.num_classes))
        for batch_i, i in enumerate(batch_indices):
            batch_labels[batch_i] = self.labels[i]

        # Find list of IDs and pass in for feature extraction
        batch_files  = [self.files[k] for k in batch_indices]
        X = self.__extract_features(batch_files)
        
        # Deal with missing data in the X array
        X, batch_labels = self.__treat_missing_data(X, batch_labels)
        return X, batch_labels

    def on_epoch_end(self):
        """Updates indices after each epoch"""

        self.indices = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __extract_features(self, batch_files):
        """
            Generates data containing samples for a given batch.  Differing 
            from the original Keras VGGish implementation, each item in the
            batch_files list here represents a segment of a wave file, so
            they don't need to be further sub-divided.
            
            Return format is based on pretrained Keras VGGish model input shape.
            
            Returns X : np.array (num_samples, num_frames, num_bands, 1)
            
            IMPORTANT: if any samples were too small to be processed, there
            will be rows in the returned X array which are filled with zeros.
        """
        
        num_files = len(batch_files)
        X = np.zeros((num_files, mel_params.NUM_FRAMES, mel_params.NUM_BANDS, 1))
        
        # Generate data from the appropriate segment of the audio file
        for i, file in enumerate(batch_files):
            data, sample_rate = sf.read(file.split(':')[0], 
                                        start = int(file.split(':')[1]), 
                                        frames = int(file.split(':')[2]))
            # Transform to log mel spectrogram format and store sample
            spectrogram = _waveform_to_mel_spectrogram_segments(data, sample_rate)
            spectrogram = np.expand_dims(spectrogram, 3)
            
            # anticipate case where sound sample was too small to create the spectrogram
            if spectrogram.shape[0] > 0:
                X[i, :, :, :] = spectrogram

        return X
    
    def __treat_missing_data(self, X, batch_files):
        """
            Check for rows in X that contain all zeros (indicating audio sample
            was too short), and remove those rows from both X and batch_files
            arrays, thus reducing the batch size.
        """

#         print('DEBUG: np.count_nonzero (ax=0)={}'.format(np.count_nonzero(X, axis=0)))
#         if np.count_nonzero(X, axis=0) != self.batch_size:
#             pprint.pprint('DEBUG: np.nonzero(X)={}'.format(np.nonzero(X)[0]))

        return X, batch_files