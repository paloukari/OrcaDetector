# -*- coding: future_fstrings -*-

"""
VGGish model for Keras. A VGG-like model for audio classification

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton

Adapted from https://github.com/DTaoo/VGGish
"""

import numpy as np
import sys

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K

# project-specific imports
import orca_params
import mel_params

class OrcaVGGish(object):    
    """
    An adaption of the VGGish architecture for our audio classification model.
    This model is never trained if weights aren't provided!
    """
    
    def __init__ (self,
                  load_weights=True, 
                  weights='audioset', 
                  input_tensor=None,
                  input_shape=None,
                  out_dim=None,
                  pooling='avg'):

        """
        :param load_weights: if load weights
        :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
        :param input_tensor: input_layer
        :param input_shape: input data shape
        :param out_dim: output dimension
        :param pooling: pooling type over the non-top network, 'avg' or 'max'

        :return: A Keras model instance.
        """

        # Validate parameters
        if weights not in {'audioset', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `audioset` '
                             '(pre-training on audioset).')

        out_dim = orca_params.NUM_CLASSES

        if input_shape is None:
            input_shape = (mel_params.NUM_FRAMES, mel_params.NUM_BANDS, 1)  # 496, 64, 1

        # VGGish model was trained with a "batch-first" matrix format.
        if input_tensor is None:
            audio_input = Input(shape=input_shape, name='input_1')
        else:
            if not K.is_keras_tensor(input_tensor):
                audio_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
            else:
                audio_input = input_tensor

        # TODO: determine if we need a BatchNormalization layer to process input before
        #   feeding to the pretrained VGGish layers.
        
        # Build VGGish model
        # Block 1
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(audio_input)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

        # Block 2
        x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

        # Block 3
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)

        # FC block (this is the space for our custom adaptations of the VGGish model)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='orca_pool4')(x)
        x = Flatten(name='orca_flatten_')(x)
        x = Dense(4096, activation='relu', name='orca_fc1/orca_fc1_1')(x)
        x = Dense(4096, activation='relu', name='orca_fc1/orca_fc1_2')(x)
        x = Dense(out_dim, activation='softmax', name='orca_softmax')(x)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = audio_input

        # Instantiate model
        self.model = Model(inputs=inputs, outputs=x, name='OrcaVGGish')

        # load weights
        if load_weights:
            if weights == 'audioset':
                print('Pretrained weights will be loaded from {}'.format(orca_params.WEIGHTS_PATH))
                self.model.load_weights(orca_params.WEIGHTS_PATH, by_name=True)
            else:
                raise Exception("ERROR: failed to load weights")
        
        # print representation of the model
        self.model.summary()

        # Build and compile the model
        print('Compiling model with {} optimizer and {} loss.' \
              .format(orca_params.OPTIMIZER, orca_params.LOSS))
        self.model.compile(optimizer=orca_params.OPTIMIZER,
                           loss=orca_params.LOSS,
                           metrics=['accuracy'])

    def get_model(self):
        
        return self.model
    

class VGGish(object):    
    """
    An implementation of the original VGGish architecture.
    This model is never trained if weights aren't provided!
    """
    
    def __init__ (self,
                  load_weights=True, 
                  weights='audioset', 
                  input_tensor=None,
                  input_shape=None,
                  out_dim=None,
                  include_top=True,
                  pooling='avg'):

        """
        :param load_weights: if load weights
        :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
        :param input_tensor: input_layer
        :param input_shape: input data shape
        :param out_dim: output dimension
        :param include_top:whether to include the 3 fully-connected layers at the top of the network.
        :param pooling: pooling type over the non-top network, 'avg' or 'max'

        :return: A Keras model instance.
        """

        # Validate parameters
        if weights not in {'audioset', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `audioset` '
                             '(pre-training on audioset).')

        if out_dim is None:
            out_dim = mel_params.EMBEDDING_SIZE  # 128

        if input_shape is None:
            input_shape = (mel_params.NUM_FRAMES, mel_params.NUM_BANDS, )  # 496, 64, [batch]

        if input_tensor is None:
            audio_input = Input(shape=input_shape, name='input_1')
        else:
            if not K.is_keras_tensor(input_tensor):
                audio_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
            else:
                audio_input = input_tensor

        # Build VGGish model

        # Block 1
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(audio_input)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

        # Block 2
        x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

        # Block 3
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)

        if include_top:
            # FC block
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
            x = Flatten(name='flatten_')(x)
            x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
            x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
            x = Dense(out_dim, activation='relu', name='vggish_fc2')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)


        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = audio_input

        # Instantiate model
        self.model = Model(inputs, x, name='VGGish')

        # load weights
        if load_weights:
            if weights == 'audioset':
                if include_top:
                    print('Weights will be loaded from {}'.format(orca_params.WEIGHTS_PATH_TOP))
                    self.model.load_weights(orca_params.WEIGHTS_PATH_TOP)
                else:
                    print('Weights will be loaded from {}'.format(orca_params.WEIGHTS_PATH))
                    self.model.load_weights(orca_params.WEIGHTS_PATH)
            else:
                raise Exception("ERROR: failed to load weights")
        
        # print representation of the model
        self.model.summary()

    def get_model(self):
        
        return self.model
    
if __name__ == '__main__':

    """
    Simple example to confirm if weights can be loaded.
    """
    print('Loading original VGGish model:')
    sound_extractor = VGGish(load_weights=True, 
                             weights='audioset',
                             include_top=False, 
                             pooling='avg').get_model()

    print('Loading OrcaVGGish model:')
    sound_extractor = OrcaVGGish(load_weights=True, 
                                 weights='audioset',
                                 pooling='avg').get_model()

    # debugging output to verify that trained weights were loaded
    layers = sound_extractor.layers
    for l in layers:
        print('layer={}'.format(l.name))
        for w in l.get_weights():
            print('  weights shape={}'.format(w.shape))
            print('  weights max={}\n'.format(np.max(w)))
        
    print('Done!')