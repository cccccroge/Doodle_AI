import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


TRAIN_PATH = 'dataset/train_png/'
IMAGE_DIM = 150
IMAGE_CLASS = 3

KERNEL_SIZE = 3
BATCH_SIZE = 20
EPOCHS = 100
VALIDATION_SPLIT = 0.2

def get_model():
    model = Sequential([
        Conv2D( 32, (KERNEL_SIZE, KERNEL_SIZE), 
                activation='relu', input_shape=(IMAGE_DIM, IMAGE_DIM, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), activation='relu'),
        MaxPooling2D(2, 2),

        Dropout(0.5),
        Flatten(),
        
        Dense(512, activation='relu'),
        Dense(IMAGE_CLASS, activation='softmax')
    ])

    model.compile(  loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['acc', ] )
    model.summary()

    return model

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMAGE_DIM, IMAGE_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


classifier = get_model()
train_generator, validation_generator = get_generators()
history = classifier.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples,
    verbose=1
)
classifier.save('model/classifier_3.h5')