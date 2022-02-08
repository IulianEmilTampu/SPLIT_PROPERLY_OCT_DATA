'''
The model implemented here is the LigthOCT model proposed by
Butola et al., in 2020. DOI https://doi.org/10.1364/BOE.395487.
'''

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dense, Flatten

# custom imports
import utilities_models_tf

##
class LightOCT(object):
    '''
    Implementation of the LightOCT described in https://arxiv.org/abs/1812.02487
    used for OCT image classification.
    The model architecture is:
    conv(5x5, 8 filters) - ReLU - MaxPool(2x2) - conv(5x5, 32) - ReLU - Flatten - Softmax - outpul Layer
    '''
    def __init__(self, number_of_input_channels,
                    num_classes,
                    input_size,
                    data_augmentation=True,
                    normalizer=None,
                    class_weights=None,
                    kernel_size=(5,5),
                    model_name='LightOCT',
                    debug=False):

        self.number_of_input_channels = number_of_input_channels
        self.num_classes = num_classes
        self.input_size=input_size
        self.debug = debug
        if class_weights is None:
            self.class_weights = np.ones([1, self.num_classes])
        else:
            self.class_weights = class_weights
        self.model_name = model_name
        self.kernel_size = kernel_size

        inputs = Input(shape=[input_size[0], input_size[1], self.number_of_input_channels])

        # augmentation
        if data_augmentation:
            x = utilities_models_tf.augmentor(inputs)
        else:
            x = inputs

        # building LightOCT model
        x = Conv2D(filters=8,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        padding='same',
                        )(x)
        x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        x = Conv2D(filters=32,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        padding='same',
                        )(x)
        x = MaxPooling2D(pool_size=(2,2),
                        strides=2
                        )(x)
        # FCN
        x = Flatten()(x)
        final = Dense(units=self.num_classes, activation='softmax')(x)

        # save model paramenters
        self.num_filter_start = 8
        self.depth = 2
        self.num_filter_per_layer = [8, 32]
        self.custom_model = False

        # finally make the model and return
        self.model = Model(inputs=inputs, outputs=final, name=model_name)

        # print model if needed
        if self.debug is True:
            print(self.model.summary())
