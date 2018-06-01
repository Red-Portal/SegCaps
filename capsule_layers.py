
'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde, 
Modified by: Red-Portal, removed several Keras dependencies
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of the various capsule layers and dynamic routing and squashing functions.
'''

import tensorflow as tf
import numpy as np

def deconv_length(dim_size, stride_size, kernel_size, padding):
    if dim_size is None:
        return None
    if padding == 'VALID':
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == 'FULL':
        dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
    elif padding == 'SAME':
        dim_size = dim_size * stride_size
    return dim_size

def batch_flatten(input_tensor):
    flat_shape = tf.stack([-1, tf.reduce_prod(tf.shape(input_tensor)[1:])])
    return tf.reshape(input_tensor, flat_shape)

def capsule_length(inputs, name="length", keepdims=False):
    with tf.variable_scope(name):
        if inputs.get_shape().ndims == 5:
            assert inputs.get_shape()[-2].value == 1, \
                'Error: Must have num_capsules = 1 going into Length'
            inputs = tf.squeeze(inputs, axis=-2)
        return tf.norm(inputs, axis=-1, keepdims=keepdims)

def quantize_onehot(inputs, axis=1, name="quantize_onehot"):
    with tf.variable_scope(name):
        return tf.one_hot(indices=tf.argmax(x, axis),
                          num_classes=x.get_shape().as_list()[axis])

def mask(inputs, mask=None, name="mask"):
    if mask == None:
        mask = quantize_onehot(inputs)
    masked = batch_flatten(mask * input)
    return mask * input

def conv2d_capsule(inputs, kernel_size, num_capsules, num_atoms,
                   strides=1, padding='SAME', routings=3,
                   initializer=tf.initializers.random_normal,
                   name="conv2d_capsule"):

    assert len(inputs.shape) == 5, \
        "The input Tensor should have shape= "\
        "[None, height, width, num_capsule, num_atoms]"

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    in_capsules = inputs.shape[3]
    in_atoms = inputs.shape[4]
    with tf.variable_scope(name):
        W = tf.get_variable('weight', [kernel_size, kernel_size, in_atoms, num_capsules * num_atoms],
                            initializer=initializer, dtype=tf.float32)
        b = tf.get_variable('bias', [1, 1, num_capsules, num_atoms],
                            initializer=tf.initializers.constant(0.1), dtype=tf.float32)

        inputs = tf.transpose(inputs, [3, 0, 1, 2, 4])
        inputs = tf.reshape(inputs, [batch_size * in_capsules, in_height, in_width, in_atoms])
        inputs.set_shape((None, in_height, in_width, in_atoms))

        inputs = tf.nn.conv2d(inputs, W, (1,strides, strides,1), padding=padding, data_format='NHWC')
        votes_shape = tf.shape(inputs)
        out_height, out_width = inputs.shape[1:3]

        inputs = tf.reshape(inputs, [batch_size, in_capsules, out_height,
                                     out_width, num_capsules, num_atoms])
        inputs.set_shape((None, in_capsules, out_height, out_width, num_capsules, num_atoms))

        logit_shape = [batch_size, in_capsules, out_height, out_width, num_capsules]
        b = tf.tile(b, [out_height, out_width, 1, 1])

        activations = update_routing(votes=inputs, biases=b, logit_shape=logit_shape,
                                     num_dims=6, input_dim=in_capsules,
                                     output_dim=num_capsules, num_routing=routings)
    return activations

def conv2d_transpose_capsule(inputs, kernel_size, num_capsules, num_atoms, scaling=2,
                             upsamp_type='deconv', padding='SAME', routings=3,
                             initializer=tf.initializers.random_normal,
                             name="conv2d_transpose_capsule"):
    assert len(inputs.shape) == 5, \
        "The input Tensor should have shape= "\
        "[None, height, width, num_capsule, num_atoms]"

    input_shape = tf.shape(inputs)
    batch_size = tf.shape(inputs)[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    in_capsules = inputs.shape[3]
    in_atoms = inputs.shape[4]

    with tf.variable_scope(name):
        if upsamp_type == 'subpix':
            weight_shape=[kernel_size, kernel_size, in_atoms,
                          num_capsules * num_atoms * scaling * scaling]
        elif upsamp_type == 'resize':
            weight_shape=[kernel_size, kernel_size,
                          in_atoms, num_capsules * num_atoms]
        elif upsamp_type == 'deconv':
            weight_shape=[kernel_size, kernel_size,
                          num_capsules * num_atoms, in_atoms]
        else:
            raise NotImplementedError('Upsampling must be one of: \
            "deconv", "resize", or "subpix"')

        W = tf.get_variable(shape=weight_shape, initializer=initializer, name='weight')
        b = tf.get_variable(shape=[1, 1, num_capsule, num_atoms],
                            initializer=tf.initializers.constant(0.1), name='bias')

        inputs = tf.transpose(inputs, [3, 0, 1, 2, 4])
        input_shape = tf.shape(input_transposed)
        inputs = tf.reshape(input_transposed, [batch_size * in_capsules, in_height, in_width, in_atoms])
        inputs.set_shape((None, in_height, in_width, in_atoms))

        if self.upsamp_type == 'resize':
            inputs = tf.image.resize_images(inputs, scaling, scaling, 'NHWC')
            inputs = tf.nn.conv2d(inputs, kernel=W, strides=(1,1,1,1),
                                  padding=padding, data_format='NHWC')
        elif self.upsamp_type == 'subpix':
            inputs = tf.nn.conv2d(inputs, kernel=W, strides=(1,1,1,1), padding='same', data_format='NHWC')
            inputs = tf.depth_to_space(inputs, scaling)
        else:
            batch_size = batch_size * in_capsules
            out_height = deconv_length(in_height, scaling, kernel_size, padding)
            out_width = deconv_length(in_width, scaling, kernel_size, padding)
            output_shape = (batch_size, out_height, out_width, num_capsule * num_atoms)
            outputs = tf.nn.conv2d_transpose(inputs, W, output_shape, (1,self.scaling, self.scaling,1),
                                             padding=padding, data_format='NHWC')
        votes_shape = tf.shape(inputs)
        out_height, out_width = inputs.shape[1:3]

        inputs = tf.reshape(inputs, [batch_size, in_capsules, out_height,
                                     out_width, num_capsules, num_atoms])
        inputs.set_shape((None, in_capsules, out_height, out_width, num_capsules, num_atoms))

        logit_shape = [batch_size, in_capsules, out_height, out_width, num_capsules]
        b = tf.tile(b, [out_height, out_width, 1, 1])

        activations = update_routing(votes=inputs, biases=b, logit_shape=logit_shape,
                                     num_dims=6, input_dim=in_capsules,
                                     output_dim=num_capsules, num_routing=routings)
    return activations

def update_routing(votes, biases, logit_shape, num_dims,
                   input_dim, output_dim, num_routing):
    if num_dims == 6:
        votes_t_shape = [5, 0, 1, 2, 3, 4]
        r_t_shape = [1, 2, 3, 4, 5, 0]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    height, width, caps = votes_trans.shape[3:]

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits, dim=-1)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash(preactivate)
        activations = activations.write(i, activation)
        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        #print(logits.shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=-1)
        #print(votes.shape, " ", act_replicated.shape)
        logits += distances
        #print(logits.shape)
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
      dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
      lambda i, logits, activations: i < num_routing,
      _body,
      loop_vars=[i, logits, activations],
      swap_memory=True)

    return tf.cast(activations.read(num_routing - 1), dtype='float32')


def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))
