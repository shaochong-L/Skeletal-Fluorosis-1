#Stage 2
#The segmentation result and the original image are fused and classified

from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Reshape, MaxPooling3D, multiply, InputSpec, SeparableConv2D, advanced_activations
from keras.layers import add, Flatten, Subtract,multiply
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import matplotlib
import numpy as np
import cv2
from PIL import Image
from keras.callbacks import TensorBoard
matplotlib.use('Agg')


# Global Constants
NB_CLASS=3
IM_WIDTH=768
IM_HEIGHT=1024
train_root=     'train/'
vaildation_root='vaildation/'
test_root=      'test/'
model_root =    'model.h5'
best_model_root ='model_best.h5'

batch_size = 12
EPOCH = 100

# train data
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(IM_HEIGHT, IM_WIDTH),
    batch_size=batch_size,
    shuffle=True
)

# vaild data
vaild_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255
)
vaild_generator = train_datagen.flow_from_directory(
    vaildation_root,
    target_size=(IM_HEIGHT, IM_WIDTH),
    batch_size=batch_size,
)

# test data
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = train_datagen.flow_from_directory(
    test_root,
    target_size=(IM_HEIGHT, IM_WIDTH),
    batch_size=batch_size,
)

class KMax(Layer):
    def __init__(self,**kwargs):
        super(KMax,self).__init__(**kwargs)
        self.k = 24

    def call(self, inputs):
        x = tf.nn.top_k(inputs, self.k, sorted=True, name=None)[0]
        return x 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k)

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x
def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Dropout(0.3)(x)
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
def Dense6(inputs, num_filter, growrate):
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = concatenate([inputs, conv1], axis=-1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = concatenate([conv1, conv2], axis=-1)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = concatenate([conv2, conv3], axis=-1)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = concatenate([conv3, conv4], axis=-1)

    conv5 = BatchNormalization()(conv4)
    conv5 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = concatenate([conv4, conv5], axis=-1)

    conv6 = BatchNormalization()(conv5)
    conv6 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = concatenate([conv5, conv6], axis=-1)

    outputs = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv6)
    return outputs
def Dense12(inputs, num_filter, growrate):
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = concatenate([inputs, conv1], axis=-1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = concatenate([conv1, conv2], axis=-1)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = concatenate([conv2, conv3], axis=-1)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = concatenate([conv3, conv4], axis=-1)

    conv5 = BatchNormalization()(conv4)
    conv5 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = concatenate([conv4, conv5], axis=-1)

    conv6 = BatchNormalization()(conv5)
    conv6 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = concatenate([conv5, conv6], axis=-1)

    conv7 = BatchNormalization()(conv6)
    conv7 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = concatenate([conv6, conv7], axis=-1)

    conv8 = BatchNormalization()(conv7)
    conv8 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = concatenate([conv7, conv8], axis=-1)

    conv9 = BatchNormalization()(conv8)
    conv9 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = concatenate([conv8, conv9], axis=-1)

    conv10 = BatchNormalization()(conv9)
    conv10 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv10)
    conv10 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = concatenate([conv9, conv10], axis=-1)

    conv11 = BatchNormalization()(conv10)
    conv11 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv11)
    conv11 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv11)
    conv11 = concatenate([conv10, conv11], axis=-1)

    conv12 = BatchNormalization()(conv11)
    conv12 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = concatenate([conv11, conv12], axis=-1)

    outputs = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv12)
    return outputs
def Dense24(inputs, num_filter, growrate):
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = concatenate([inputs, conv1], axis=-1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = concatenate([conv1, conv2], axis=-1)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = concatenate([conv2, conv3], axis=-1)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = concatenate([conv3, conv4], axis=-1)

    conv5 = BatchNormalization()(conv4)
    conv5 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = concatenate([conv4, conv5], axis=-1)

    conv6 = BatchNormalization()(conv5)
    conv6 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = concatenate([conv5, conv6], axis=-1)

    conv7 = BatchNormalization()(conv6)
    conv7 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = concatenate([conv6, conv7], axis=-1)

    conv8 = BatchNormalization()(conv7)
    conv8 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = concatenate([conv7, conv8], axis=-1)

    conv9 = BatchNormalization()(conv8)
    conv9 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = concatenate([conv8, conv9], axis=-1)

    conv10 = BatchNormalization()(conv9)
    conv10 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv10)
    conv10 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = concatenate([conv9, conv10], axis=-1)

    conv11 = BatchNormalization()(conv10)
    conv11 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv11)
    conv11 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv11)
    conv11 = concatenate([conv10, conv11], axis=-1)

    conv12 = BatchNormalization()(conv11)
    conv12 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = concatenate([conv11, conv12], axis=-1)

    conv13 = BatchNormalization()(conv12)
    conv13 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv13)
    conv13 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv13)
    conv13 = concatenate([conv12, conv13], axis=-1)

    conv14 = BatchNormalization()(conv13)
    conv14 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv14)
    conv14 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv14)
    conv14 = concatenate([conv13, conv14], axis=-1)

    conv15 = BatchNormalization()(conv14)
    conv15 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv15)
    conv15 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv15)
    conv15 = concatenate([conv14, conv15], axis=-1)

    conv16 = BatchNormalization()(conv15)
    conv16 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv16)
    conv16 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv16)
    conv16 = concatenate([conv15, conv16], axis=-1)

    conv17 = BatchNormalization()(conv16)
    conv17 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv17)
    conv17 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv17)
    conv17 = concatenate([conv16, conv17], axis=-1)

    conv18 = BatchNormalization()(conv17)
    conv18 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv18)
    conv18 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv18)
    conv18 = concatenate([conv17, conv18], axis=-1)

    conv19 = BatchNormalization()(conv18)
    conv19 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv19)
    conv19 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv19)
    conv19 = concatenate([conv18, conv19], axis=-1)

    conv20 = BatchNormalization()(conv19)
    conv20 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv20)
    conv20 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv20)
    conv20 = concatenate([conv19, conv20], axis=-1)

    conv21 = BatchNormalization()(conv20)
    conv21 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv21)
    conv21 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv21)
    conv21 = concatenate([conv20, conv21], axis=-1)

    conv22 = BatchNormalization()(conv21)
    conv22 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv22)
    conv22 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv22)
    conv22 = concatenate([conv21, conv22], axis=-1)

    conv23 = BatchNormalization()(conv22)
    conv23 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv23)
    conv23 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv23)
    conv23 = concatenate([conv22, conv23], axis=-1)

    conv24 = BatchNormalization()(conv23)
    conv24 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv24)
    conv24 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv24)
    conv24 = concatenate([conv23, conv24], axis=-1)

    outputs = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv24)
    return outputs
def Dense16(inputs, num_filter, growrate):
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = concatenate([inputs, conv1], axis=-1)

    conv2 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = concatenate([conv1, conv2], axis=-1)

    conv3 = BatchNormalization()(conv2)
    conv3 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = concatenate([conv2, conv3], axis=-1)

    conv4 = BatchNormalization()(conv3)
    conv4 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = concatenate([conv3, conv4], axis=-1)

    conv5 = BatchNormalization()(conv4)
    conv5 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = concatenate([conv4, conv5], axis=-1)

    conv6 = BatchNormalization()(conv5)
    conv6 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = concatenate([conv5, conv6], axis=-1)

    conv7 = BatchNormalization()(conv6)
    conv7 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = concatenate([conv6, conv7], axis=-1)

    conv8 = BatchNormalization()(conv7)
    conv8 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = concatenate([conv7, conv8], axis=-1)

    conv9 = BatchNormalization()(conv8)
    conv9 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = concatenate([conv8, conv9], axis=-1)

    conv10 = BatchNormalization()(conv9)
    conv10 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv10)
    conv10 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv10)
    conv10 = concatenate([conv9, conv10], axis=-1)

    conv11 = BatchNormalization()(conv10)
    conv11 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv11)
    conv11 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv11)
    conv11 = concatenate([conv10, conv11], axis=-1)

    conv12 = BatchNormalization()(conv11)
    conv12 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv12)
    conv12 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv12)
    conv12 = concatenate([conv11, conv12], axis=-1)

    conv13 = BatchNormalization()(conv12)
    conv13 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv13)
    conv13 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv13)
    conv13 = concatenate([conv12, conv13], axis=-1)

    conv14 = BatchNormalization()(conv13)
    conv14 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv14)
    conv14 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv14)
    conv14 = concatenate([conv13, conv14], axis=-1)

    conv15 = BatchNormalization()(conv14)
    conv15 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv15)
    conv15 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv15)
    conv15 = concatenate([conv14, conv15], axis=-1)

    conv16 = BatchNormalization()(conv15)
    conv16 = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv16)
    conv16 = Conv2D(growrate, (3, 3), activation='relu', padding='same')(conv16)
    conv16 = concatenate([conv15, conv16], axis=-1)

    outputs = Conv2D(num_filter, (1, 1), activation='relu', padding='same')(conv16)
    return outputs
def Sep(input, channels, keranls = 3):
    input = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(input)
    conv1 = SeparableConv2D(channels, keranls, padding="same", data_format="channels_last", activation='relu')(input)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    conv2 = SeparableConv2D(channels, keranls, padding="same", data_format="channels_last", activation='relu')(conv1)
    return conv2
def Ens_L(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    input1 = Lambda(lambda x : x[:, 0:512, :, :], name="Lambda_x1")(inpt)
    input2 = Lambda(lambda x : x[:, 512:1024, :, :], name="Lambda_x2")(inpt)

    # input
    x1 = Conv2d_BN(input1, nb_filter=8, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x1)
    x2 = Conv2d_BN(input2, nb_filter=8, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x2)
    x_in = add([x1, x2])
    #x1 = Conv2D(8, (1, 1))(x1)
    #x_in = multiply([x1, x_in])


    # ResNet
    res = identity_Block(x_in, nb_filter=8, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=8, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=8, kernel_size=(3, 3))

    res = identity_Block(res, nb_filter=16, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=16, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=16, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=16, kernel_size=(3, 3))

    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=32, kernel_size=(3, 3))

    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))
    res = identity_Block(res, nb_filter=64, kernel_size=(3, 3))

    res = Conv2D(1, (3, 3), padding='same', activation='relu')(res)
    res = Flatten()(res)
    res = KMax()(res)

    # DenseNet
    dense = Dense6(x_in, 8, 2)

    dense = MaxPooling2D(pool_size=(2, 2))(dense)
    dense = Dense12(dense, 16, 2)

    dense = MaxPooling2D(pool_size=(2, 2))(dense)
    dense = Dense24(dense, 32, 2)

    dense = MaxPooling2D(pool_size=(2, 2))(dense)
    dense = Dense16(dense, 64, 2)

    dense = Conv2D(1, (3, 3), padding='same', activation='relu')(dense)
    dense = Flatten()(dense)
    dense = KMax()(dense)



    # Sep
    sep1 = Sep(x_in, 8)
    sep1 = Sep(sep1, 8)
    sep1 = Sep(sep1, 8)

    sep2 = MaxPooling2D(pool_size=(2, 2))(sep1)
    sep2 = Sep(sep2, 16)
    sep2 = Sep(sep2, 16)
    sep2 = Sep(sep2, 16)
    sep2 = Sep(sep2, 16)

    sep3 = MaxPooling2D(pool_size=(2, 2))(sep2)
    sep3 = Sep(sep3, 32)
    sep3 = Sep(sep3, 32)
    sep3 = Sep(sep3, 32)
    sep3 = Sep(sep3, 32)
    sep3 = Sep(sep3, 32)
    sep3 = Sep(sep3, 32)

    sep4 = MaxPooling2D(pool_size=(2, 2))(sep3)
    sep4 = Sep(sep4, 64)
    sep4 = Sep(sep4, 64)
    sep4 = Sep(sep4, 64)

    sep4 = Conv2D(1, (3, 3), padding='same', activation='relu')(sep4)
    sep4 = Flatten()(sep4)
    sep4 = KMax()(sep4)

    conv = concatenate([dense, res, sep4], axis=-1)

    x = Dense(classes, activation='softmax')(conv)
    model = Model(inputs=inpt, outputs=x)

    return model

def my_loss(y_true, y_pred):
    y_true = K.argmax(y_true, axis=1)
    loss = 0
    for i in range(batch_size):
        y_score = y_pred[i][y_true[i]]
        loss += -K.log(y_score + 1e-6)
    loss = loss/16.
    return loss
def learn_rate(epoch):
    lr = 0.1
    if epoch > 2:
        lr = 0.03
    if epoch > 4:
        lr = 0.01
    if epoch > 6:
        lr = 0.003
    if epoch > 10:
        lr = 0.001
    if epoch > 15:
        lr = 0.0003
    if epoch > 30:
        lr = 0.0001
    if epoch > 60:
        lr = 0.00001
    return lr

def check_print():
    model = Ens_L(IM_HEIGHT,IM_WIDTH,3,NB_CLASS)
    model.summary()
    plot_model(model, to_file='model.png')
    model.compile(optimizer='adam', loss=my_loss,metrics=['acc'])
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc',top_k_categorical_accuracy])
    print('Model Compiled')
    return model
def get_inputs(src=[]):
        pre_x = []
        for s in src:
            input = cv2.imread(s)
            input = cv2.resize(input, (IM_WIDTH, IM_HEIGHT))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            pre_x.append(input)  # input一张图片
        pre_x = np.array(pre_x) / 255.0
        return pre_x
def get_one_image(image):
        pre_x = []
        input = cv2.imread(image)
        input = cv2.resize(input, (IM_WIDTH, IM_HEIGHT))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)
        pre_x = np.array(pre_x) / 255.0
        return pre_x

lr = LearningRateScheduler(learn_rate)

if __name__ == '__main__':
    if not os.path.exists(model_root):
        model=check_print()
        checkpoint = ModelCheckpoint(best_model_root, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit_generator(train_generator, validation_data=vaild_generator, epochs=EPOCH,
                            steps_per_epoch=train_generator.n / batch_size
                            , validation_steps=vaild_generator.n / batch_size,
                            callbacks=callbacks_list)
        model.save(model_root)
        
    model = load_model(model_root, custom_objects={'my_loss': my_loss, 'KMax': KMax})
    model.summary()
    predict_dir = test_root
    test = os.listdir(predict_dir)
    print(test)
    images = []
    for testpath in test:
        for fn in os.listdir(os.path.join(predict_dir, testpath)):
            if fn.endswith('png'):
                fd = os.path.join(predict_dir, testpath, fn)
                pre_x = get_one_image(fd)
                pre_y = model.predict(pre_x)
                print(fn, pre_y[0])

    loss,acc=model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)
    print('Test result:loss:%f,acc:%f' % (loss, acc))
