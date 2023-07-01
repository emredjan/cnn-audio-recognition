from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
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
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.utils import normalize, plot_model, to_categorical
from sklearn.preprocessing import LabelEncoder

from cnn_audio.params import pr

# TODO: find a way to get this dynamically for the parsing function
DATA_SHAPE: tuple[int, ...] = (50, 94, 2)

def parse_tfrecord(example_proto):

    feature_description = {
        'input': tf.io.FixedLenFeature([DATA_SHAPE[0] * DATA_SHAPE[1] * DATA_SHAPE[2]], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    parsed_example['input'] = tf.reshape(parsed_example['input'], DATA_SHAPE)

    input_feature = parsed_example['input']
    label_feature = parsed_example['label']

    return input_feature, label_feature

def build_model(
    num_classes: int,
    input_shape:tuple[int, ...],
):
    input_layer = Input(shape=input_shape)

    conv_1 = Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu',
        kernel_regularizer=l2(0.001),
    )(input_layer)

    conv_2 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(conv_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropout_1 = Dropout(0.25)(max_pool_1)

    conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(
        dropout_1
    )

    conv_4 = Conv2D(128, (3, 3), activation='relu')(conv_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    dropout_2 = Dropout(0.25)(max_pool_2)

    conv_5 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(
        dropout_2
    )

    conv_6 = Conv2D(128, (3, 3), activation='relu')(conv_5)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)
    dropout_3 = Dropout(0.25)(max_pool_3)

    flatten_1 = Flatten()(dropout_3)
    dense_1 = Dense(1024, activation='relu')(flatten_1)
    dropout_4 = Dropout(0.5)(dense_1)
    output_layer = Dense(num_classes, activation='softmax')(dropout_4)

    model = Model(inputs=input_layer, outputs=output_layer)

    opt = Adam(learning_rate=1e-4, beta_1=1e-4 / pr['model']['epochs'])
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def build_modelxxx(
    num_classes: int,
    input_shape:tuple[int, ...],
):

    input_layer = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=outputs)

    opt = Adam(learning_rate=1e-4, beta_1=1e-4 / pr['model']['epochs'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



def prepare_training(run_id):
    weights_dir = Path(pr['locations']['weights_base_dir']) / run_id
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_file = str(weights_dir) + '/' + pr['model']['weight_file_pattern']
    checkpoint = ModelCheckpoint(weights_file, save_best_only=True)

    earlystopping = EarlyStopping(patience=pr['model']['early_stop'])

    log_dir = Path(pr['locations']['log_base_dir']) / run_id

    tensorboard = TensorBoard(log_dir=str(log_dir), update_freq=50)  # type: ignore

    callbacks_list = [checkpoint, earlystopping, tensorboard]

    return callbacks_list


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
