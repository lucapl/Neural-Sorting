import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, Activation, Layer, Concatenate, Lambda, Normalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import NonNeg,MinMaxNorm
from tensorflow.keras import backend as K

from src.tensors.activations import leaky_hard_sigmoid

class MonotoneBlock(Layer):
    '''
    Layer incorporating monotonic utility function
    '''
    def __init__(self, branches=3,units=1, mon_type=True, **kwargs):
        assert branches >= 2
        super(MonotoneBlock, self).__init__(**kwargs)
        self.units = units
        self.branches = branches
        self.mon_type = mon_type

    def build(self, input_shape):
        self.sigmoids = [Dense(self.units,
                               activation=Activation(leaky_hard_sigmoid),
                               kernel_constraint=NonNeg(),bias_constraint=NonNeg()) for _ in range(self.branches)] # the y and bias for each x
        for sig in self.sigmoids:
            sig.build(input_shape)
        #self.sigmoids = Concatenate(axis=2)([sigs])
        self.concat = Concatenate()
        self.linear = Dense(self.units,activation=None,use_bias=False,kernel_constraint=NonNeg()) # alphas of each branch
        self.linear.build((None,self.branches*self.units))
        super(MonotoneBlock, self).build(input_shape)

    def call(self, inputs):
        #x = tf.math.reduce_sum([sig(inputs) for sig in self.sigmoids],axis=0)
        x = self.concat([sig(inputs) for sig in self.sigmoids])
        x = self.linear(x)
        return x
    
class Thresholder(Layer):
    def __init__(self, threshold=0.5, **kwargs):
        super(Thresholder, self).__init__(**kwargs)
        self.threshold = tf.constant(threshold)#tf.Variable(threshold)

    def call(self, inputs):
        '''
            Similarly to the example from the labs utiliy over threshold indicates belonging to class 1 and under threshold to class 0
            to ease it the threshold is substracted from the value
        '''
        # indices = tf.argmax(tf.cast(tf.less(inputs, tf.expand_dims(self.thresholds, axis=0)), tf.float32), axis=1)
        # one_hot = tf.one_hot(indices, depth=len(self.thresholds)+1)
        # return one_hot        
        return inputs - self.threshold
