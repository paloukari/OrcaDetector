import tensorflow as tf
from keras.models import Sequential
from generator import WavDataGenerator
from database_parser import index_files

def print_framework_versions():
    print(tf.VERSION)
    print(tf.keras.__version__)


def create_network():
    # TODO: Create the network
    return model


def run(**params):

    print_framework_versions()
    train_files, train_labels, validate_files, validate_labels= index_files('../data')
    
    training_generator = WavDataGenerator(
        train_files, train_labels, **params)

    # test the generator
    training_generator.__getitem__(0)

    validation_generator = WavDataGenerator(
        validate_files,  validate_labels, **params)

    model = create_network()

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=1)


if __name__ == '__main__':
    run()
