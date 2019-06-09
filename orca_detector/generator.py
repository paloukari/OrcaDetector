import numpy as np
import keras
import librosa
from vggish.mel_features import log_mel_spectrogram, frame
import soundfile as sf
import resampy
import params

def waveform_to_examples(data, sample_rate):
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != params.sample_rate:
    data = resampy.resample(data, sample_rate, params.sample_rate)

  # Compute log mel spectrogram features.
  log_mel = log_mel_spectrogram(
      data,
      audio_sample_rate=params.sample_rate,
      log_offset=params.log_offset,
      window_length_secs=params.stft_window_length_seconds,
      hop_length_secs=params.stft_hop_length_seconds,
      num_mel_bins=params.num_mel_bins,
      lower_edge_hertz=params.mel_min_hz,
      upper_edge_hertz=params.mel_max_hz )

  # Frame features into examples.
  features_sample_rate = 1.0 / params.stft_hop_length_seconds
  example_window_length = int(round(
      params.quantum_size* features_sample_rate))
  example_hop_length = int(round(
      params.quantum_hop * features_sample_rate))
  log_mel_examples = frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples


class WavDataGenerator(keras.utils.Sequence):
    'Generates data for Keras from Wav files'

    def __init__(self, files, labels, shuffle = True):
        'Initialization'
        self.files = files
        self.labels = labels
        self.n_classes = labels.shape[1]
        self.len = int(np.floor(len(self.files) / params.batch_size))
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*params.batch_size:(index+1)*params.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X = self.__extract_features(files_temp)

        return X,labels_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __extract_features(self, files_temp):
        # X : (n_samples, *(frames_size, bands_size))
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        # Generate data
        for i, file in enumerate(files_temp):
            data, sr = sf.read(
                file.split(':')[0], 
                start = int(file.split(':')[1]), 
                frames = int(file.split(':')[2]))
            # Store sample
            X.append(waveform_to_examples(data, sr))

        return X
