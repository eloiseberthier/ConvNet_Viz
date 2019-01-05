import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers import MaxPooling2D as KMaxPooling2D
 
from deconvnet.deconv2D import Deconv2D
from deconvnet.pool_unpool import MaxPooling2D, UndoMaxPooling2D

def truncated_model(model, last_layer=None):
    """
    Given a trained network, compute the equivalent
    truncated model with argmax MaxPooling2D.
    Only supports MaxPooling2D and Conv2D layers.
    """
    _, sx, sy, chan = model.layers[0].input_shape
    inp = Input(shape=(sx, sy, chan), name="input")
    x = inp
    i = 0
    positions = []
    while i < len(model.layers):
        if not type(model.layers[i]) in [KMaxPooling2D, Conv2D]:
            pass
        elif type(model.layers[i]) is KMaxPooling2D:
            x, pos = MaxPooling2D(name=model.layers[i].name)(x)
            positions.append(pos)
        elif type(model.layers[i]) is Conv2D:
            x = model.layers[i](x)
        if model.layers[i].name == last_layer:
            break
        i += 1
    return Model(inputs=inp, outputs=[x]+positions)

def backward_network(model):
    """
    Given a truncated network containing only Conv2D
    and custom MaxPooling2D layers, compute the reverse
    deconvnet. Only supports MaxPooling2D and Conv2D layers.
    """
    inputs = []
    if type(model.output) is list:
        for j, out in enumerate(model.output):
            _,  a, b, c = out.shape.as_list()
            inp = Input(batch_shape=(1, a, b, c), name="input_{}".format(j+1))
            inputs.append(inp)
    else: # There is only one output (no MaxPooling2D)
        _,  a, b, c = model.output.shape.as_list()
        inputs = [Input(batch_shape=(1, a, b, c), name="input_1")]
    x = inputs[0]
    i = len(model.layers)-1 # Layer count
    k = len(inputs)-1 # MaxPool count
    while i > 0:
        if type(model.layers[i]) is MaxPooling2D:
            _, a, b, c = model.layers[i].input_shape
            l = UndoMaxPooling2D(out_shape=(1, a, b, c),
                                 name=model.layers[i].name)
            x = l([x, inputs[k]])
            k -= 1
        elif type(model.layers[i]) is Conv2D:
            w, b = model.layers[i].weights
            kx, ky, chan_in, chan_out = w.shape.as_list()
            l = Deconv2D(chan_in, (kx, ky), activation='relu',
                         padding=model.layers[i].padding,
                         name=model.layers[i].name)
            x = l(x)
            l.set_weights(model.layers[i].get_weights())
        i -= 1
    return Model(inputs=inputs, outputs=x)