from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    MaxPooling2D,
    SpatialDropout2D,
)
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop, SGD, Nadam, Adagrad
from keras.regularizers import l2
from keras.utils import normalize, plot_model, to_categorical
from sklearn.preprocessing import LabelEncoder

from cnn_audio.params import pr

AUDIO_FEATURES = pr['model']['features']
DATA_SHAPE = (50, 94, len(AUDIO_FEATURES))
VALUE_COUNT = 50 * 94 * len(AUDIO_FEATURES)


def parse_tfrecord(example_proto):
    # TODO: find a way to get these dynamically (and efficiently!) for the parsing function
    data_shape: tuple[int, ...] = DATA_SHAPE  # (50, 94, 1)
    value_count = VALUE_COUNT  # 4700  # data_shape[0] * data_shape[1] * data_shape[2]

    feature_description = {
        'input': tf.io.FixedLenFeature([value_count], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    parsed_example['input'] = tf.reshape(parsed_example['input'], data_shape)

    input_feature = parsed_example['input']
    label_feature = parsed_example['label']

    return input_feature, label_feature


def build_model_0(
    num_classes: int,
    input_shape: tuple[int, ...],
):
    input_layer = Input(shape=input_shape)

    conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001),)(input_layer)
    conv_2 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(conv_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropout_1 = Dropout(0.25)(max_pool_1)

    conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(dropout_1)
    conv_4 = Conv2D(128, (3, 3), activation='relu')(conv_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    dropout_2 = Dropout(0.25)(max_pool_2)

    conv_5 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(dropout_2)
    conv_6 = Conv2D(128, (3, 3), activation='relu')(conv_5)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)
    dropout_3 = Dropout(0.25)(max_pool_3)

    flatten_1 = Flatten()(dropout_3)
    dense_1 = Dense(1024, activation='relu')(flatten_1)
    dropout_4 = Dropout(0.25)(dense_1)
    output_layer = Dense(num_classes, activation='softmax')(dropout_4)

    model = Model(inputs=input_layer, outputs=output_layer)

    # opt = AdamW(learning_rate=1e-4, epsilon=1e-1 / pr['model']['epochs'])
    opt = Adagrad(learning_rate=0.01, epsilon=1e-7)
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    description = 'base model'

    return model, description


def build_model_1(
    num_classes: int,
    input_shape: tuple[int, ...],
):
    input_layer = Input(shape=input_shape)

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Fourth convolutional layer
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    # Change the kernel size to (2, 2) to avoid the error
    pool4 = MaxPooling2D((2, 2), padding='same')(conv4)

    # Flatten the output of the convolutional layers
    flattened = Flatten()(pool4)

    # Dropout layer
    dropout1 = Dropout(0.2)(flattened)

    # Fully connected layer
    output = Dense(num_classes, activation='softmax')(dropout1)

    model = Model(input_layer, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    description = 'bard provided model'

    return model, description


def build_model_2(
    num_classes: int,
    input_shape: tuple[int, ...],
):
    input_layer = Input(shape=input_shape)

    # Add convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the output and add dense layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    description = 'simple model'

    return model, description


def prepare_training(run_id):
    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    weights_dir = Path(pr['locations']['weights_base_dir']) / run_id
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_file = str(weights_dir) + '/' + pr['model']['weight_file_pattern']
    checkpoint = ModelCheckpoint(weights_file, save_best_only=True)

    lr_scheduler = LearningRateScheduler(scheduler)
    earlystopping = EarlyStopping(patience=pr['model']['early_stop'])

    log_dir = Path(pr['locations']['log_base_dir']) / run_id

    tensorboard = TensorBoard(log_dir=str(log_dir), update_freq=50)  # type: ignore

    callbacks = [checkpoint, earlystopping, tensorboard, lr_scheduler]

    return callbacks


def train_model(
    model: Model,
    data_train: tf.data.Dataset,
    data_valid: tf.data.Dataset,
    callbacks: list,
    epochs: int,
    batch_size: int,
):
    fitted_model = model.fit(
        x=data_train,
        y=None,  # Embedded in tf dataset
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=data_valid,
        verbose=1,  # type: ignore
    )

    return fitted_model


def save_model(model, file_name):
    model.save(file_name, overwrite=True, save_format=None)
