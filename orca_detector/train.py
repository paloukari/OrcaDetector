import tensorflow as tf
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from generator import WavDataGenerator

from database_parser import index_files

def print_framework_versions():
    print(tf.VERSION)
    print(tf.keras.__version__)


def create_network():
    # 1 channel, maybe 1-sec audio signal, for an example.
    input_shape = (1, 44100)
    sr = 44100
    model = Sequential()
    # A mel-spectrogram layer
    model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                             padding='same', sr=sr, n_mels=128,
                             fmin=0.0, fmax=sr/2, power_melgram=1.0,
                             return_decibel_melgram=False, trainable_fb=False,
                             trainable_kernel=False,
                             name='trainable_stft'))
    # Maybe some additive white noise.
    model.add(AdditiveNoise(power=0.2))
    # If you wanna normalise it per-frequency
    # or 'channel', 'time', 'batch', 'data_sample'
    model.add(Normalization2D(str_axis='freq'))
    # After this, it's just a usual keras workflow. For example..
    # Add some layers, e.g., model.add(some convolution layers..)
    # Compile the model
    # if single-label classification
    model.compile('adam', 'categorical_crossentropy')
    # train it with raw audio sample inputs

    return model


def run(**params):

    print_framework_versions()
    train_files, train_labels, validate_files, validate_labels= index_files('../data')
    
    training_generator = WavDataGenerator(
        train_files, train_labels, **params)

    training_generator.__getitem__(0)

    validation_generator = WavDataGenerator(
        validate_files,  encoder.transform(validate_labels), **params)

    model = create_network()

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=1)


if __name__ == '__main__':
    run()
