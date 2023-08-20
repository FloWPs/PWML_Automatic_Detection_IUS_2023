#######################
#  IMPORT LIBRAIRIES  #
#######################


import time as time


from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K
from keras import Input, Model
from keras.models import load_model
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling2D, Flatten, Dropout, ELU, Layer, Concatenate



########################
#  MODEL ARCHITECTURE  #
########################

class WeightedAverage(Layer):

    def __init__(self, n_output):
        super(WeightedAverage, self).__init__()
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=[1,1,n_output], minval=0, maxval=1),
            trainable=True) # (1,1,n_inputs)

    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1) # (n_batch, n_feat)


def new_model(WINDOW=32, channels=3):

    input = Input((WINDOW, WINDOW, channels))
    inputS = input[:,:,:,0:1]
    inputC = input[:,:,:,1:2]


    xS = Conv2D(filters= 64, kernel_size=(5, 5),  strides=(1, 1),
               padding="same", kernel_initializer="glorot_uniform", name="1_convS")(inputS)
    xS = ELU(alpha=.5, name='1_conveluS')(xS)
    xS = MaxPooling2D(pool_size=(2, 2), name="1_maxS")(xS)

    xC = Conv2D(filters= 64, kernel_size=(5, 5),  strides=(1, 1),
               padding="same", kernel_initializer="glorot_uniform", name="1_convC")(inputC)
    xC = ELU(alpha=.5, name='1_conveluC')(xC)
    xC = MaxPooling2D(pool_size=(2, 2), name="1_maxC")(xC)

    xS = Conv2D(128, (5, 5), padding="same", name="2_convS")(xS)
    xS = ELU(alpha=.5, name='2_conveluS')(xS)
    xS = MaxPooling2D(pool_size=(2, 2), name="2_maxS")(xS)

    xC = Conv2D(128, (5, 5), padding="same", name="2_convC")(xC)
    xC = ELU(alpha=.5, name='2_conveluC')(xC)
    xC = MaxPooling2D(pool_size=(2, 2), name="2_maxC")(xC)

    xS = Conv2D(256, (5, 5), padding="same", activation='relu', name="3_convS")(xS)
    xS = MaxPooling2D(pool_size=(2, 2), name="3_maxS")(xS)

    xC = Conv2D(256, (5, 5), padding="same", activation='relu', name="3_convC")(xC)
    xC = MaxPooling2D(pool_size=(2, 2), name="3_maxC")(xC)

    xS = Conv2D(512, (5, 5), padding="same", activation='relu', name="4_convS")(xS)
    xS = Dropout(.35, name='4_dropoutS')(xS)
    xS = MaxPooling2D(pool_size=(2, 2), name="4_maxS")(xS)

    xC = Conv2D(512, (5, 5), padding="same", activation='relu', name="4_convC")(xC)
    xC = Dropout(.35, name='4_dropoutC')(xC)
    xC = MaxPooling2D(pool_size=(2, 2), name="4_maxC")(xC)

    # FUSION LAYER
    xS = Flatten()(xS)
    xC = Flatten()(xC)
    
    x = [xS, xC]
    x = WeightedAverage(n_output=len(x))(x)

    x = Dense(units=256, activation='relu', kernel_initializer="glorot_uniform")(x)
    output_tensor = Dense(2, activation="softmax")(x)

    ########################### ########################### ###########################

    model = tf.keras.models.Model(input, output_tensor, name='new_model')
    return model
