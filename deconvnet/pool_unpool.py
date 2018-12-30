from keras.engine.topology import Layer

import numpy as np

import tensorflow as tf

class MaxPooling2D(Layer):
    def __init__(self, pool_size = 2, stride = None, padding = 'VALID', **kwargs):
        self.pool_size = pool_size
        assert(isinstance(self.pool_size, int))
        self.stride = stride
        if self.stride is None:
            self.stride = self.pool_size
        assert(isinstance(self.stride, int))
        self.padding = padding
        assert(padding in ['VALID', 'SAME'])
        super(MaxPooling2D, self).__init__(**kwargs)
            
    def build(self, input_shape):
        super(MaxPooling2D, self).build(input_shape)

    def call(self, inp):
        out, pos = tf.nn.max_pool_with_argmax(inp, 
                            ksize = [1, self.pool_size, self.pool_size, 1],
                            strides = [1, self.stride, self.stride, 1],
                            padding = self.padding)
        return [out, pos]

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.padding == 'VALID':
            output_shape[1] = output_shape[1] - self.pool_size + 1
            output_shape[2] = output_shape[2] - self.pool_size + 1
        output_shape[1] = (output_shape[1] + self.stride - 1) // self.stride
        output_shape[2] = (output_shape[2] + self.stride - 1) // self.stride
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

class UndoMaxPooling2D(Layer):
    def __init__(self, out_shape, **kwargs):
        self.out_shape = out_shape
        assert(isinstance(self.out_shape, tuple))
        assert(len(self.out_shape) == 4)
        super(UndoMaxPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UndoMaxPooling2D, self).build(input_shape)

    def call(self, inp):
        x, pos = inp
        pos = tf.cast(pos, dtype = tf.int32)
        x = tf.reshape(x, [-1])
        pos = tf.reshape(pos, [-1])
        out = tf.Variable(tf.zeros(np.prod(self.out_shape)))
        out = tf.scatter_update(out, pos, x)
        return tf.reshape(out, self.out_shape)

    def compute_output_shape(self, input_shape):
        return self.out_shape