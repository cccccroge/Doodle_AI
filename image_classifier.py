import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D


FILTER_SIZE = 32
KERNEL_SIZE = 3

model = Sequential([
    Conv2D( FILTER_SIZE, (KERNEL_SIZE, KERNEL_SIZE), 
            activation='relu', input_shape=)
])