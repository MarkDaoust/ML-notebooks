from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections

import numpy as np

import tensorflow as tf

def _variable(name, shape, stddev, wd = 0.0):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev),name=name)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# Simple Nonlinearities

def ELU(x):
    return tf.nn.elu(x)

ELU.up = lambda x:ELU(x-1.0)+1.0

def softplus2(x):
    """ softplus scaled by 1/2 to look more like ELU"""
    with tf.name_scope('softplus2'):
        return tf.nn.softplus(x*2.0)/2.0

softplus2.down = lambda x:softplus2(x+1.0)-1.0

# Basic layer objects, call to specify a shape, then call with an input to the layer

class Conv(object):
    """
    Simple Conv Later
    """
    def __init__(self, shape, stddev=0.1):
        self.shape = shape

        with tf.name_scope('Conv'):
            self.kernel = _variable('weights', shape=shape,stddev=stddev)

            self.biases = _variable('biases', [shape[-1]], stddev=stddev)

    def __call__(self, images):
        with tf.name_scope('Conv'):
            conv = tf.nn.conv2d(images, self.kernel, strides = [1,1,1,1], padding='SAME')

            conv = tf.nn.bias_add(conv, self.biases)

            return conv

def pool2(x):
    with tf.name_scope('pool2'):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def global_average_pooling(x):
    with tf.name_scope('global_average_pooling'):
        return tf.reduce_mean(x,[1,2])

class Linear(object):
    """
    Simple Linear Layer
    """
    def __init__(self, shape, stddev=0.1):
        with tf.name_scope('Linear'):
            with tf.name_scope('weight'):
                self.weights = _variable('weights', shape=shape, stddev=stddev)

            with tf.name_scope('bias'):
                self.biases = _variable('biases', shape = [shape[1]], stddev=stddev)

    def __call__(self,input):
        with tf.name_scope('Linear'):
            out = tf.matmul(input, self.weights) + self.biases
            return out


class Chain(object):
    """simple function composition"""
    def __init__(self,items,scope=None):
        self.scope = scope
        if self.scope is None:
            self.items = items
        else:
            with tf.name_scope(scope):
                self.items = items

    def __call__(self,arg):
        if self.scope is None:
            result = self._call(arg)
        else:
            with tf.name_scope(self.scope):
                result = self._call(arg)

        return result

    def _call(self,arg):
        for fun in self.items:
            arg = fun(arg)

        return arg


class PReLU(object):
    """trainable generalization of [leaky-ReLU]
    max(x,a*x+b)"""
    def __init__(self, shape, scope="PReLU"):
        self.scope = scope
        with tf.name_scope(scope):
            self.slope = tf.Variable(0.1+tf.zeros(shape,tf.float32))

    def __call__(self,x):
        with tf.name_scope(self.scope):
            return tf.maximum(x,self.slope*x)


class Bilinear(object):
    """trainable generalization of [ReLU, leaky-ReLU, PRelu, max(x,-1)]
       max(x,a*x+b)"""
    def __init__(self,shape,scope="bilinear"):
        self.scope = scope
        with tf.name_scope(scope):
            self.bias = tf.Variable(tf.zeros(shape,tf.float32)-1)
            self.slope = tf.Variable(tf.zeros(shape,tf.float32))

    def __call__(self,x):
        with tf.name_scope(self.scope):
            return tf.maximum(x,self.slope*x+self.bias)

class Maxout(object):
    """simple maxout-2 layer"""
    def __init__(self,f1,f2,scope='maxout'):
        self.scope = scope
        with tf.name_scope(scope):
            self.f1 = f1
            self.f2 = f2

    def __call__(self,x):
        with tf.name_scope(self.scope):
            y1 = self.f1(x)
            y2 = self.f2(x)

            return tf.maximum(y1,y2)

#parens in dir-names is a bad idea
simple = collections.OrderedDict([
    ['linear',lambda x:x],
    ['tanh',tf.tanh],
    #standard ReLU
    ['ReLU',lambda x:tf.maximum(x,0)],
    #a shifted ReLU
    ['ReLU.down',lambda x:tf.maximum(x,-1)],
    #ReLU.shift(0.0) == softplus2(0.0), same positive bias as softplus
    ['ReLU.shift',lambda x:tf.maximum(x+np.log(1+np.exp(2*0))/2,0)],
    ['LReLU',lambda x:tf.maximum(x,0.1*x)],
    #the proposed ELU
    ['ELU',ELU],
    #ELU with the asymptote at Zero
    ['ELU.up',ELU.up],
    #softplus
    ['softplus2',softplus2],
    #a shifted softplus with asymptote at -1
    ['softplus2.down',softplus2.down],
])

def plot(names=None):
    '''
    plot the simple nonlinearities
    '''

    import matplotlib.pyplot as plt

    if names is None:
        names = simple.keys()

    x_np=np.linspace(-2,2,1001,dtype=np.float32)
    x = tf.constant(x_np)

    ys=[]
    labels=[]
    with tf.Session() as sess:
        for label in names:
            nl = simple[label]
            y = nl(x)
            ys.append(y.eval())
            labels.append(label)
    sess.close()

    A = plt.axes()
    A.grid(True)

    ys = np.array(ys).T

    lines=plt.plot(x_np,ys)
    plt.legend(lines, names, loc='upper left')


# once initialized, these can all be called with a shape as input.
# the result is some sort of layer object, with all it's trainable variables instantiated
# (so that the "train" and "validate" stacks can share the variables)
# that layer can in turn be called with an argument for the layer to process
class SimpleFactory(object):
    def __init__(self,scope,base,nonlinearity):
        self.scope = scope
        self.base = base
        self.nonlinearity = nonlinearity

    def __call__(self,shape,**kwargs):
        with tf.name_scope(self.scope):
            return Chain([
                self.base(shape,**kwargs),
                self.nonlinearity,
            ],scope = self.scope)

class BilinearFactory(object):
    def __init__(self,scope,base):
        self.scope = scope
        self.base = base

    def __call__(self,shape,**kwargs):
        with tf.name_scope(self.scope):
            return Chain([
                self.base(shape,**kwargs),
                Bilinear(shape[-1:]),
            ],scope=self.scope)

class PreluFactory(object):
    def __init__(self,scope,base):
        self.scope = scope
        self.base = base

    def __call__(self,shape,**kwargs):
        with tf.name_scope(self.scope):
            return Chain([
                self.base(shape,**kwargs),
                PReLU(shape[-1:]),
            ],scope=self.scope)

class MaxoutFactory(object):
    def __init__(self,scope,base):
        self.scope = scope
        self.base = base

    def __call__(self,shape,**kwargs):
        with tf.name_scope(self.scope):
            return Maxout(
                self.base(shape,**kwargs),
                self.base(shape,**kwargs),scope=self.scope)



fcs = collections.OrderedDict(
    [key,SimpleFactory('fc_'+key,Linear,fun)] for (key,fun) in simple.iteritems())

fcs['bilinear'] = BilinearFactory('fc_Bilinear',Linear)
fcs['maxout'] = MaxoutFactory('fc_Maxout',Linear)
fcs['PReLU'] = PreluFactory('fc_PReLU',Linear)


convs = collections.OrderedDict(
    [key,SimpleFactory('covn_'+key,Conv,fun)] for (key,fun) in simple.iteritems())

convs['bilinear'] = BilinearFactory('conv_Bilinear',Conv)
convs['maxout'] = MaxoutFactory('conv_Maxout',Conv)
convs['PReLU'] = PreluFactory('conv_PReLU',Conv)

NAMES = convs.keys()