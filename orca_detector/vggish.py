"""
VGGish model for Keras. A VGG-like model for audio classification

# Reference

- [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html), ICASSP 2017
- VGGish original code: https://github.com/tensorflow/models/tree/master/research/audioset
- Keras version: https://github.com/DTaoo/VGGish

"""

import sys

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K

# project-specific imports
import vggish_params as params

# weight path; when Docker container is run, weights path on the host
# machine is expected to be mapped to /weights
WEIGHTS_PATH = '/weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = '/weights/vggish_audioset_weights.h5'

class VGGish(object):    
    """
    An implementation of the VGGish architecture.
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
            out_dim = params.EMBEDDING_SIZE  # 128

        if input_shape is None:
            input_shape = (params.NUM_FRAMES, params.NUM_BANDS, 1)  # 496, 64

        if input_tensor is None:
            aud_input = Input(shape=input_shape, name='input_1')
        else:
            if not K.is_keras_tensor(input_tensor):
                aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
            else:
                aud_input = input_tensor

        # Build VGGish model

        # Block 1
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
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
            inputs = aud_input

        # Instantiate model
        self.model = Model(inputs, x, name='VGGish')

        # load weights
        if load_weights:
            if weights == 'audioset':
                if include_top:
                    print('Weights will be loaded from {}'.format(WEIGHTS_PATH_TOP))
                    self.model.load_weights(WEIGHTS_PATH_TOP)
                else:
                    print('Weights will be loaded from {}'.format(WEIGHTS_PATH))
                    self.model.load_weights(WEIGHTS_PATH)
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
    sound_extractor = VGGish(load_weights=True, 
                             weights='audioset',
                             include_top=False, 
                             pooling='avg').get_model()

    print('Done!')