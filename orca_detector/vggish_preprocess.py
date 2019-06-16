"""
VGGish model for Keras. A VGG-like model for audio classification
  
# Reference

- [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017
- VGGish original code: https://github.com/tensorflow/models/tree/master/research/audioset
- Keras version: https://github.com/DTaoo/VGGish

"""

import numpy as np
import resampy

# project-specific imports
import mel_features
import vggish_params as params

def preprocess_sound(data, sample_rate):
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
        data = np.mean(data, axis=1)

    # Resample to the rate assumed by VGGish.
    if sample_rate != params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(data,
                                               audio_sample_rate=params.SAMPLE_RATE,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=params.MEL_MIN_HZ,
                                               upper_edge_hertz=params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate  = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length    = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples      = mel_features.frame(log_mel,
                                               window_length=example_window_length,
                                               hop_length=example_hop_length)
    
    return log_mel_examples


