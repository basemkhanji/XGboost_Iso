import os

import tensorflow as tf
from tensorflow import keras

new_model = keras.models.load_model('my_model.h5')
new_model.summary()

