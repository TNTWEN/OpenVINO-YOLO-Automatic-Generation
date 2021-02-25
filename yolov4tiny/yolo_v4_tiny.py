# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo_v4 import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORSTINY = [(10, 14),  (23, 27),  (37, 58),
            (81, 82),  (135, 169),  (344, 319)]
_ANCHORS = [(12, 16), (19, 36), (40, 28),
                (36, 75), (76, 55), (72, 146),
                (142, 110), (192, 243), (459, 401)]

def _tiny_res_block(inputs,in_channels,channel1,channel2,channel3,data_format):
    net = _conv2d_fixed_padding(inputs,in_channels,kernel_size=3)

    route = net
    #_,split=tf.split(net,num_or_size_splits=2,axis=1 if data_format =="NCHW" else 3)
    split = net[:, in_channels//2:, :, :]if data_format=="NCHW" else net[:, :, :, in_channels//2:]
    net = _conv2d_fixed_padding(split,channel1,kernel_size=3)
    route1 = net
    net = _conv2d_fixed_padding(net,channel2,kernel_size=3)
    net = tf.concat([net, route1], axis=1 if data_format == 'NCHW' else 3)
    net = _conv2d_fixed_padding(net,channel3,kernel_size=1)
    feat = net
    net = tf.concat([route, net], axis=1 if data_format == 'NCHW' else 3)
    net = slim.max_pool2d(
        net, [2, 2], scope='pool2')
    return net,feat




def yolo_v4_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v4 tiny model.
    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v4-tiny'):
                    # paste output of parse_config.py here

                    return detections