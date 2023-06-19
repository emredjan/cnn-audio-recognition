from pathlib import Path
from typing import Tuple

import click
import joblib
import numpy as np
import pandas as pd
import pendulum

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
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.utils import normalize, plot_model, to_categorical
from sklearn.preprocessing import LabelEncoder

from cnn_audio.params import pr

TARGETS: list[str] = pr['training']['target']
RUNID: str = pendulum.now().format('YYYYMMDD_HHmmss')


def combine_columns(*args):
    return '_'.join(args)


def encode_classes(metadata_paths: list[Path]):
    df = pd.read_json(metadata_paths[0], orient='index')

    for mdp in metadata_paths[1:]:
        metadata = pd.read_json(mdp, orient='index')
        df = pd.concat([df, metadata], axis=0, join='outer')

    target: pd.Series = df[TARGETS].astype(str).apply(lambda row: combine_columns(*row), axis=1)

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

    target: pd.Series = df[TARGETS].astype(str).apply(lambda row: combine_columns(*row), axis=1)
    target_enc: np.ndarray = encoder.transform(target)  # type: ignore

    X: np.ndarray = data.reshape(data.shape + (1,))
    y: np.ndarray = to_categorical(target_enc, len(encoder.classes_))

    return X, y


def build_model(
    num_classes: int,
    input_shapes: list[tuple[int]],
):
    input_images = []
    for input_shape in input_shapes:
        input_images.append(Input(shape=input_shape))

    if len(input_shapes) > 1:
        input_layer = Concatenate(axis=-1)(input_images)
    else:
        input_layer = input_images[0]

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

    model = Model(inputs=input_images, outputs=output_layer)

    opt = Adam(learning_rate=1e-4, beta_1=1e-4 / pr['training']['epochs'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    plot_model(model, to_file='./model.png', show_shapes=True)

    return model


def prepare_training():

    weights_dir = Path(pr['locations']['weights_base_dir']) / RUNID
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_file = str(weights_dir) + '/' + pr['training']['weight_file_pattern']
    checkpoint = ModelCheckpoint(weights_file, save_best_only=True)

    earlystopping = EarlyStopping(patience=pr['training']['early_stop'])

    log_dir = Path(pr['locations']['log_base_dir']) / RUNID

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
    epochs: int,
    batch_size: int,
):
    fitted_model = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )

    return fitted_model
