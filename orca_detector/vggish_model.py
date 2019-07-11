# -*- coding: future_fstrings -*-

"""
VGGish model for Keras. A VGG-like model for audio classification

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from https://github.com/DTaoo/VGGish
"""

import numpy as np
import os
import sys

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, \
    Dropout
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras import optimizers

# project-specific imports
import mel_params
import orca_params


class VGGish(object):
    """
    An implementation of the original VGGish architecture.
    This model is never trained if weights aren't provided!
    Model structure and weights are from https://github.com/DTaoo/VGGish
    """

    def custom_preprocessing_layers(self):
        """
            Applies custom layers before passing the input into VGGish layers.
            This can be used in subclasses for BatchNormalization, for example.
        """
        self.x = self.audio_input

    def _build_vggish_base_layers(self):
        """
            Builds the original VGGish base layers so that weights can be loaded.
            DO NOT RENAME LAYERS or weights will not be loaded.
        """

        # Block 1
        self.x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv1')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                              name='pool1')(self.x)

        # Block 2
        self.x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv2')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2),
                              padding='same', name='pool2')(self.x)

        # Block 3
        self.x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv3/conv3_1')(self.x)
        self.x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv3/conv3_2')(self.x)
        self.x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                              name='pool3')(self.x)

        # Block 4
        self.x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv4/conv4_1')(self.x)
        self.x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu',
                        padding='same', name='conv4/conv4_2')(self.x)

    def _build_vggish_top_layers(self):
        """
            Builds the original VGGish top layers so that weights can be loaded.
            DO NOT RENAME LAYERS or weights will not be loaded.
        """
        # FC block
        self.x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                              name='pool4')(self.x)
        self.x = Flatten(name='flatten_')(self.x)
        self.x = Dense(4096, activation='relu',
                       name='vggish_fc1/fc1_1')(self.x)
        self.x = Dense(4096, activation='relu',
                       name='vggish_fc1/fc1_2')(self.x)
        self.x = Dense(self.out_dim, activation='relu',
                       name='vggish_fc2')(self.x)

    def custom_top_layers(self):
        """
            Applies custom layers after the VGGish layers.  This method needs to
            be replaced with a customer implementation in subclasses used for
            different classification tasks.
        """
        if self.pooling == 'avg':
            self.x = GlobalAveragePooling2D()(self.x)
        elif self.pooling == 'max':
            self.x = GlobalMaxPooling2D()(self.x)

    def __init__(self,
                 load_weights=True,
                 weights='audioset',
                 input_tensor=None,
                 input_shape=None,
                 out_dim=None,
                 include_top=True,
                 pooling='avg',
                 optimizer='adam',
                 lr=0.001,  # default for Adam
                 loss='categorical_crossentropy',
                 dropout=0.,
                 model_name='VGGish'):
        """
            Args:
                load_weights = boolean if weights should be loaded
                weights = weights to load ('audioset' weights are pre-trained on YouTube-8M').
                input_tensor = Keras input_layer
                input_shape = input data shape
                out_dim = output dimension
                include_top = boolean whether to include the final 3 fully-connected layers
                pooling = pooling type over the non-top network ('avg' or 'max')
                optimizer = string name of Keras optimizer
                loss = string name of Keras loss function

            Returns:
                A compiled Keras model instance
        """

        if out_dim is None:
            out_dim = mel_params.EMBEDDING_SIZE  # 128
        self.out_dim = out_dim

        if input_shape is None:
            input_shape = (mel_params.NUM_FRAMES,
                           mel_params.NUM_BANDS, 1)  # 496, 64, 1
        self.input_shape = input_shape

        # VGGish model was trained with a "batch-first" matrix format.
        if input_tensor is None:
            audio_input = Input(shape=input_shape, name='input_1')
        else:
            if not K.is_keras_tensor(input_tensor):
                audio_input = Input(tensor=input_tensor,
                                    shape=input_shape, name='input_1')
            else:
                audio_input = input_tensor
        self.audio_input = audio_input

        self.pooling = pooling
        self.dropout = dropout

        # Build model.  Subclasses should implement the custom_top_layers() method
        # and optionally, the custom_preprocessing_layers() method.
        self.custom_preprocessing_layers()

        self._build_vggish_base_layers()

        if include_top:
            self._build_vggish_top_layers()
        else:
            self.custom_top_layers()

        if input_tensor is not None:
            self.inputs = get_source_inputs(input_tensor)
        else:
            self.inputs = audio_input

        # Instantiate model
        self.model = Model(inputs=self.inputs, outputs=self.x, name=model_name)

        # load weights
        if load_weights:
            # Use audioset weights for initial training
            if weights == 'audioset':
                if include_top:
                    print('Loading weights from {}'.format(
                        (orca_params.WEIGHTS_PATH_TOP)))
                    self.model.load_weights(
                        orca_params.WEIGHTS_PATH_TOP, by_name=True)
                else:
                    print('Loading weights from {}'.format(
                        (orca_params.WEIGHTS_PATH)))
                    if not os.path.exists(orca_params.WEIGHTS_PATH):
                        raise Exception(
                            'ERROR: cannot find {} to load.'.format((orca_params)))
                    self.model.load_weights(
                        orca_params.WEIGHTS_PATH, by_name=True)
            # Use our own saved weights for running inference
            else:
                print(f'Loading weights from {weights}')
                if not os.path.exists(weights):
                    raise Exception('ERROR: cannot find {weights}.')
                self.model.load_weights(weights, by_name=True)
                

        # print representation of the model
        self.model.summary()

        # instantiate optimizer
        if optimizer == 'sgd':
            optimizer_obj = optimizers.SGD(lr=lr)
        elif optimizer == 'adam':
            optimizer_obj = optimizers.Adam(lr=lr)
        else:
            # Fallback to default params for others
            optimizer_obj = optimizers.get(optimizer)

        # Build and compile the model
        self.model.compile(optimizer=optimizer_obj,
                           loss=loss,
                           metrics=['accuracy'])
        print('Compiled {} model with {} optimizer (lr={}) and {} loss.'.format(
            (model_name), (optimizer), (lr), (loss)))

    def get_model(self):
        """ Returns a compiled Keras model."""
        return self.model


class OrcaVGGish(VGGish):
    """
    A subclass that adapts the VGGish architecture for our audio classification model.
    This model is never trained if weights aren't provided!
    """

    def custom_preprocessing_layers(self):
        """
            Applying batch normalization.
        """
        self.x = BatchNormalization()(self.audio_input)

    def custom_top_layers(self):
        """
            Top layers for OrcaDetector classification.
        """

        # Dropout to prevent overfitting
        self.x = Dropout(self.dropout)(self.x)

        # FC block
        self.x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                              name='orca_pool4')(self.x)
        self.x = Flatten(name='orca_flatten_')(self.x)
        self.x = Dense(4096, activation='relu',
                       name='orca_fc1/orca_fc1_1')(self.x)
        self.x = Dense(4096, activation='relu',
                       name='orca_fc1/orca_fc1_2')(self.x)
        self.x = Dense(self.out_dim, activation='softmax',
                       name='orca_softmax')(self.x)

    def __init__(self,
                 load_weights=True,
                 weights='audioset',
                 input_tensor=None,
                 input_shape=None,
                 out_dim=None,
                 pooling='avg',
                 optimizer=orca_params.OPTIMIZER,
                 lr=orca_params.LEARNING_RATE,
                 loss=orca_params.LOSS,
                 dropout=orca_params.DROPOUT,
                 model_name='OrcaVGGish'):
        """
            Args:
                load_weights = boolean if weights should be loaded
                weights = weights to load ('audioset' weights are pre-trained on YouTube-8M).
                input_tensor = Keras input_layer
                input_shape = input data shape
                out_dim = output dimension
                pooling = pooling type over the non-top network ('avg' or 'max')
                optimizer = string name of Keras optimizer
                loss = string name of Keras loss function

            Returns:
                A compiled Keras model instance
        """

        # Because we don't want to use VGGish for its original classification task,
        # we will always pass include_top=False to omit those layers.
        super().__init__(load_weights=load_weights,
                         weights=weights,
                         input_tensor=input_tensor,
                         input_shape=input_shape,
                         out_dim=out_dim,
                         include_top=False,
                         pooling=pooling,
                         optimizer=optimizer,
                         lr=lr,
                         loss=loss,
                         dropout=dropout,
                         model_name=model_name)

    def get_model(self):
        """ Returns a compiled Keras model."""
        return self.model


if __name__ == '__main__':

    """
    Simple example to confirm if weights can be loaded.
    """
    print('Loading OrcaVGGish model:')
    sound_extractor = OrcaVGGish(load_weights=True,
                                 weights='audioset',
                                 pooling='avg').get_model()

    # debugging output to verify that trained weights were loaded
    print('\nSample weights (max prior to orca_* layers should be non-zero).')
    layers = sound_extractor.layers
    for l in layers:
        print(f'layer={l.name}')
        for w in l.get_weights():
            print(f'  weights shape={w.shape}')
            print(f'  weights max={np.max(w)}\n')

    print('Done!')
