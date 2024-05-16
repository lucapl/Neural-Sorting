import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, Activation, Layer, Concatenate, Lambda, Normalization
from tensorflow.keras.constraints import NonNeg

from src.tensors.layers import Thresholder,MonotoneBlock

class NormModel(tf.keras.Model):
    def __init__(self, uta,thresh,ideal,anti,features):
        super(NormModel, self).__init__()
        self.uta = uta
        self.uta.build((features,))
        self.ideal = tf.reshape(ideal,(1,features))
        self.anti = tf.reshape(anti,(1,features))
        self.thresholdLayer = Thresholder(thresh)
        self.thresholdLayer.build(input_shape=(None,1))

    def call(self, inputs):
        out = self.uta.call(inputs)

        _min = self.uta.call(self.anti)
        _max = self.uta.call(self.ideal)

        return self.thresholdLayer((out - _min) / (_max - _min))

def create_ann_utadis_model(threshold,ideal_alt,anti_ideal_alt,n_labels,n_criteria,L=3):
    inputs = Input(shape=(n_criteria,))
    #ideal_layer = Lambda(lambda x: tf.reshape(tf.constant(ideal_alt),(1,-1)))(inputs)
    #anti_ideal_layer = Lambda(lambda x: tf.reshape(tf.constant(anti_ideal_alt),(1,-1)))(inputs)

    #concat = Concatenate(axis=0)([inputs,ideal_layer,anti_ideal_layer])

    # def splitter(x) : 
    #     split = tf.split(x, n_criteria, 1)
    #     return split    
    
    #split_layer = Lambda(splitter)(inputs)

    splits = [Lambda(lambda x: x[:, i:i+1],name=f"criteria_{i}")(inputs) for i in range(n_criteria)]

    monotones = [MonotoneBlock(branches=L,name=f"monotone_block_{i}")(split) for i,split in enumerate(splits)]
    
    concat = Concatenate(axis=1)(monotones)
    linear = Dense(1,activation=None,name="criteria_weights",use_bias=False)(concat)
    #norm = Dense(4,activation="sigmoid")(linear)

    #norm = MinMaxNormalization()(linear)

    #thresholder = Thresholder(thresholds)(norm)
    #norm = Normalization()(linear)
    model = Model(name="ann_utadis",inputs=inputs,outputs = linear) 
    
    return NormModel(model,threshold,tf.constant(ideal_alt),tf.constant(anti_ideal_alt),n_criteria)

def create_nn_model(features):
    inputs = Input((features,))
    layer = Dense(256,activation="relu")(inputs)
    layer = Dense(128,activation="relu")(inputs)
    layer = Dense(64,activation="relu")(layer)
    layer = Dense(32,activation="relu")(layer)
    outputs = Dense(1,activation="sigmoid")(layer)

    return Model(inputs=inputs,outputs=outputs)

def create_guided_model(model):
    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return tf.cast(dy > 0, dtype=tf.float32) * tf.cast(x > 0, dtype=tf.float32) * dy
        return tf.nn.relu(x), grad

    def modify_relu(layer):
        if isinstance(layer, tf.keras.layers.ReLU):
            return tf.keras.layers.Activation(guided_relu)
        return layer

    guided_model = tf.keras.models.clone_model(
        model,
        clone_function=modify_relu
    )
    guided_model.set_weights(model.get_weights())

    return guided_model