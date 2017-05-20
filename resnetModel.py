from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import decode_predictions
# from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.imagenet_utils import _obtain_input_shape
# from keras.engine.topology import get_source_inputs

from pdb import set_trace as trace

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def up_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    up_conv_name_base = 'up' + str(stage) + block + '_branch'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = UpSampling2D(size=(2,2), name=up_conv_name_base + '2a')(input_tensor)

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    

    shortcut = UpSampling2D(size=(2,2), name=up_conv_name_base + '1')(input_tensor)    
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(shortcut)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x



def ResnetModel(input_tensor=None, input_shape=None, in_channels=1, classes=1):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if input_shape is None:
        img_input = Input(shape=(in_channels,224,224))
    x = ZeroPadding2D((4, 4))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='b')
    x2 = identity_block(x, 3, [64, 64, 128], stage=2, block='c')

    x = conv_block(x2, 3, [128, 128, 256], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='b')
    #x = identity_block(x, 3, [128, 128, 256], stage=3, block='c')
    x3 = identity_block(x, 3, [128, 128, 256], stage=3, block='d')

    x = conv_block(x3, 3, [256, 256, 512], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='b')
    #x = identity_block(x, 3, [256, 256, 512], stage=4, block='c')
    #x = identity_block(x, 3, [256, 256, 512], stage=4, block='d')
    #x = identity_block(x, 3, [256, 256, 512], stage=4, block='e')
    x4 = identity_block(x, 3, [256, 256, 512], stage=4, block='f')

    x = conv_block(x4, 3, [512, 512, 1024], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 1024], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 1024], stage=5, block='c')

    
    
    x = up_conv_block(x, 3, [1024, 512, 512], stage=6, block='a')
    x = identity_block(x, 3, [1024, 512, 512], stage=6, block='b')
    x = identity_block(x, 3, [1024, 512, 512], stage=6, block='c')
    
    x = concatenate([x,x4],axis=bn_axis)

    x = up_conv_block(x, 3, [1024, 256, 256], stage=7, block='a')
    x = identity_block(x, 3, [1024, 256, 256], stage=7, block='b')
    #x = identity_block(x, 3, [1024, 256, 256], stage=7, block='c')
    #x = identity_block(x, 3, [1024, 256, 256], stage=7, block='d')
    #x = identity_block(x, 3, [1024, 256, 256], stage=7, block='e')
    x = identity_block(x, 3, [1024, 256, 256], stage=7, block='f')

    x = concatenate([x,x3],axis=bn_axis)

    x = up_conv_block(x, 3, [512, 128, 128], stage=8, block='a')
    x = identity_block(x, 3, [512, 128, 128], stage=8, block='b')
    #x = identity_block(x, 3, [512, 128, 128], stage=8, block='c')
    x = identity_block(x, 3, [512, 128, 128], stage=8, block='d')

    x = concatenate([x,x2],axis=bn_axis)        


    x = up_conv_block(x, 3, [256, 64, 64], stage=10, block='a', strides=(1, 1))
    x = identity_block(x, 3, [256, 64, 64], stage=10, block='b')
    x = identity_block(x, 3, [256, 64, 64], stage=10, block='c')

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(classes, (3, 3), padding='same', activation = 'sigmoid', name='convLast')(x)

    model = Model(img_input, x, name='resnetUnet')

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

