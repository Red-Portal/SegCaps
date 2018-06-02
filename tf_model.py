'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

import tensorflow as tf

from tf_layers import *

def CapsNetR3(inputs):
    # Layer 1: Just a conventional Conv2D layer
    inputs = tf.layers.conv2d(inputs, filters=16, kernel_size=5, strides=1,
                              padding='same', activation=tf.nn.relu, name='conv1')
    
    # inputs = tf.expand_dims(inputs, axis=3)

    # primary_caps = conv2d_capsule(inputs, kernel_size=5, num_capsules=2, num_atoms=16,
    #                               strides=2, routings=1, name='primarycaps')

    # conv_cap_2_1 = conv2d_capsule(primary_caps, kernel_size=5, num_capsules=4, num_atoms=16,
    #                               strides=1, routings=3, name='conv_cap_2_1')
    # conv_cap_2_2 = conv2d_capsule(conv_cap_2_1, kernel_size=5, num_capsules=4, num_atoms=32,
    #                               strides=2, routings=3, name='conv_cap_2_2')

    # conv_cap_3_1 = conv2d_capsule(conv_cap_2_2, kernel_size=5, num_capsules=8, num_atoms=32,
    #                               strides=1, routings=3, name='conv_cap_3_1')
    # conv_cap_3_2 = conv2d_capsule(conv_cap_3_1, kernel_size=5, num_capsules=8, num_atoms=64,
    #                               strides=2, routings=3, name='conv_cap_3_2')
    
    # conv_cap_4_1 = conv2d_capsule(conv_cap_3_2, kernel_size=5, num_capsules=8, num_atoms=32,
    #                               strides=1, routings=3, name='conv_cap_4_1')
    # deconv_cap_1_1 = conv2d_transpose_capsule(conv_cap_4_1, kernel_size=4, num_capsules=8,
    #                                          num_atoms=32, upsamp_type='deconv', scaling=2,
    #                                          routings=3, name='deconv_cap_1_1')

    # up_1 = tf.concat([deconv_cap_1_1, conv_cap_3_1], axis=-2, name='up_1',)
    # deconv_cap_1_2 = conv2d_capsule(up_1, kernel_size=5, num_capsules=4, num_atoms=32, strides=1,
    #                                 routings=3, name='deconv_cap_1_2')
    # deconv_cap_2_1 = conv2d_transpose_capsule(deconv_cap_1_2, kernel_size=4, num_capsules=4,
    #                                           num_atoms=16, upsamp_type='deconv', scaling=2,
    #                                           routings=3, name='deconv_cap_2_1')

    # up_2 = tf.concat([deconv_cap_2_1, conv_cap_2_1], axis=-2, name='up_2')
    # deconv_cap_2_2 = conv2d_capsule(up_2, kernel_size=5, num_capsules=4, num_atoms=16, strides=1,
    #                                 routings=3, name='deconv_cap_2_2')
    # deconv_cap_3_1 = conv2d_transpose_capsule(deconv_cap_2_2, kernel_size=4, num_capsules=2,
    #                                           num_atoms=16, upsamp_type='deconv', scaling=2,
    #                                           routings=3, name='deconv_cap_3_1')

    # up_3 = tf.concat([deconv_cap_3_1, inputs], axis=-2, name='up_3')

    # seg_caps = conv2d_capsule(up_3, kernel_size=1, num_capsules=1, num_atoms=16,
    #                           strides=1, routings=3, name='seg_caps')

    # out_seg = capsule_length(inputs=seg_caps, name='out_seg', keepdims=True)
    inputs = tf.layers.conv2d(inputs, filters=32, kernel_size=[3,3],
                              padding="same", activation=tf.nn.relu, name="1")
    inputs = tf.layers.conv2d(inputs, filters=32, kernel_size=[3,3],
                              padding="same", activation=tf.nn.relu, name="2")
    inputs = tf.layers.conv2d(inputs, filters=32, kernel_size=[3,3],
                              padding="same", activation=tf.nn.relu, name="3")
    inputs = tf.layers.conv2d(inputs, filters=1, kernel_size=[3,3],
                              padding="same", activation=tf.sigmoid, name="4")
    inputs = tf.squeeze(inputs)
    return inputs
