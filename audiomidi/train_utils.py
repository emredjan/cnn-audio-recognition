from pathlib import Path
from typing import Tuple

import click
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    SpatialDropout2D,
    Concatenate,
    InputLayer,
    Input,
)
from keras.models import Sequential, Model
from keras.utils import normalize, to_categorical, plot_model
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import LabelEncoder

from audiomidi import params


def encode_classes(metadata_paths):
    df = pd.read_json(metadata_paths[0], orient='index')

    for mdp in metadata_paths[1:]:
        metadata = pd.read_json(mdp, orient='index')
        df = pd.concat([df, metadata], axis=0, join='outer')

    target = df['instrument_family_str'] + '_' + df['pitch'].astype('str')
    # target = df['pitch']

    encoder = LabelEncoder()
    encoder.fit(target)

    return encoder


def prepare_data(
    data_path: Path, names_path: Path, metadata_path: Path, encoder: LabelEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    data: np.ndarray = joblib.load(data_path)
    names: np.ndarray = joblib.load(names_path)
    metadata: pd.DataFrame = pd.read_json(metadata_path, orient='index')

    df = pd.DataFrame({}, index=names).merge(
        metadata, how='left', left_index=True, right_index=True
    )
    target = df['instrument_family_str'] + '_' + df['pitch'].astype('str')
    # target: pd.Series = df['pitch']
    target_enc: np.ndarray = encoder.transform(target)  # type: ignore

    X: np.ndarray = data.reshape(data.shape + (1,))
    y: np.ndarray = to_categorical(target_enc, len(encoder.classes_))

    return X, y


def build_model_old(input_shape: Tuple[int, ...], num_classes: int) -> Sequential:
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(SpatialDropout2D(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy']
    )

    return model


def build_model_seq(input_shape, num_classes, input_shape_add=None):
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape, name="first_image"))

    if input_shape_add:
        model.add(InputLayer(input_shape=input_shape_add, name="second_image"))
        model.add(Concatenate())

    model.add(
        Conv2D(
            64,
            (3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=l2(0.001),
            # input_shape=input_shape,
        )
    )
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)
        )
    )
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)
        )
    )
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam(learning_rate=1e-4, beta_1=1e-4 / params.epochs)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def build_model(
    num_classes: int,
    input_shape: Tuple[int, ...],
    input_shape_add: Tuple[int, ...] | None = None,
):
    first_image = Input(shape=input_shape)
    input_layer = first_image
    model_inputs = first_image

    if input_shape_add:
        second_image = Input(shape=input_shape_add)
        input_layer = Concatenate(axis=-1)([first_image, second_image])
        model_inputs = [first_image, second_image]

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

    conv_3 = Conv2D(
        128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)
    )(dropout_1)

    conv_4 = Conv2D(128, (3, 3), activation='relu')(conv_3)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    dropout_2 = Dropout(0.25)(max_pool_2)

    conv_5 = Conv2D(
        128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)
    )(dropout_2)

    conv_6 = Conv2D(128, (3, 3), activation='relu')(conv_5)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)
    dropout_3 = Dropout(0.25)(max_pool_3)

    flatten_1 = Flatten()(dropout_3)
    dense_1 = Dense(1024, activation='relu')(flatten_1)
    dropout_4 = Dropout(0.5)(dense_1)
    output_layer = Dense(num_classes, activation='softmax')(dropout_4)

    model = Model(inputs=model_inputs, outputs=output_layer)

    opt = Adam(learning_rate=1e-4, beta_1=1e-4 / params.epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    plot_model(model, to_file='./model.png', show_shapes=True)

    return model


def prepare_training():
    weights_file = str(params.weights_dir) + '/' + params.weight_file_pattern
    checkpoint = ModelCheckpoint(weights_file, save_best_only=True)

    earlystopping = EarlyStopping(patience=params.early_stop)

    log_dir = params.log_dir
    try:
        for log in log_dir.rglob('events.out.*'):
            log.unlink()
        click.secho('Previous logs cleared.', fg='bright_yellow')
    except Exception:
        click.secho('Error cleaning previous logs.', fg='bright_red')

    tensorboard = TensorBoard(log_dir=str(log_dir), update_freq=50)  # type: ignore
    click.secho('Log dir: ' + str(log_dir.absolute()), fg='bright_yellow')

    callbacks_list = [checkpoint, earlystopping, tensorboard]

    return callbacks_list


def train_model(
    model,
    X_train: np.ndarray | list,
    y_train: np.ndarray,
    X_valid: np.ndarray | list,
    y_valid: np.ndarray,
    callbacks: list,
):
    fitted_model = model.fit(
        X_train,
        y_train,
        epochs=params.epochs,
        batch_size=params.batch_size,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )

    return fitted_model
