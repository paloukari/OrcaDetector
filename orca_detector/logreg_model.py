# -*- coding: future_fstrings -*-

"""
Basic LogisticRegression model for audio classification

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import numpy as np
import os
import sys

from keras.engine.topology import get_source_inputs
from keras.layers import Flatten, Dense, Input, BatchNormalization
from keras.models import Model
# from keras import backend as K
from keras import optimizers

# project-specific imports
import mel_params
import orca_params


class OrcaLogReg(object):
    """
    An implementation of LogisticRegression
    """

    def __init__(self,
                 load_weights=True,
                 weights=None,
                 out_dim=None,
                 optimizer=orca_params.LOGREG_OPTIMIZER,
                 lr=orca_params.LOGREG_LEARNING_RATE,
                 loss='categorical_crossentropy',
                 model_name='OrcaLogReg'):
        """
            Args:
                load_weights = boolean if weights should be loaded
                weights = weights to load ('audioset' weights are pre-trained on YouTube-8M').
                out_dim = output dimension
                pooling = pooling type over the non-top network ('avg' or 'max')
                optimizer = string name of Keras optimizer
                loss = string name of Keras loss function

            Returns:
                A compiled Keras model instance
        """

        if out_dim is None:
            out_dim = mel_params.EMBEDDING_SIZE  # 128
        self.out_dim = out_dim

        self.input_shape = (mel_params.NUM_FRAMES,
                            mel_params.NUM_BANDS, 1)  # 96, 64, 1
        print(f'DEBUG: self.input_shape={self.input_shape}')
        self.audio_input = Input(shape=self.input_shape, name='input_1')

        
        self.x = BatchNormalization()(self.audio_input)
        self.x = Flatten(name='orca_flatten_')(self.x)
        self.x = Dense(self.out_dim, activation='softmax',
                       name='orca_softmax')(self.x)

        self.inputs = self.audio_input
        
        # Instantiate model
        self.model = Model(inputs=self.inputs, outputs=self.x, name=model_name)

        if load_weights:
            # Use our own saved weights for running inference
            print(f'Loading weights from {weights}')
            if not os.path.exists(weights):
                raise Exception(f'ERROR: cannot find {weights}.')
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


if __name__ == '__main__':

    """
    Simple example to confirm if weights can be loaded.
    """
    print('Loading OrcaLogReg model:')
    sound_extractor = OrcaLogReg(load_weights=False).get_model()

    print('Done!')