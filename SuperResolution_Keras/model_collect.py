# -*- coding: utf-8 -*-
"""
Sor, just a homework of class.
"""
import numpy as np
import tensorflow as tf
from keras.layers import Add, Conv2D, Input, Lambda

"""
https://github.com/krasserm/super-resolution/blob/master/model/edsr.py

"""
#def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
#    x_in = Input(shape=(None, None, 3))
#    x = Lambda(normalize)(x_in)
#
#    x = b = Conv2D(num_filters, 3, padding='same')(x)
#    for i in range(num_res_blocks):
#        b = res_block(b, num_filters, res_block_scaling)
#    b = Conv2D(num_filters, 3, padding='same')(b)
#    x = Add()([x, b])
#
#    x = upsample(x, scale, num_filters)
#    x = Conv2D(3, 3, padding='same')(x)
#
#    x = Lambda(denormalize)(x)
#    return Model(x_in, x, name="edsr")

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(subpixel_conv2d(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
"""
https://github.com/krasserm/super-resolution/blob/master/model/common.py

"""
# ---------------------------------------
#  Normalization
# ---------------------------------------

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean

# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
