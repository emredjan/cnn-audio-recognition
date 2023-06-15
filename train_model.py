import os
from pathlib import Path
from typing import Tuple

import click
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.client import device_lib  # type: ignore

from audiomidi import params, train_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LABELS_FILE: Path = params.features_dir / 'label_encoder.joblib'

def encoder_export():

    metadata_paths = [
        params.train_metadata_path, params.valid_metadata_path,
        params.test_metadata_path
    ]

    click.secho('Encoding classes..', fg='bright_white', nl=False)

    try:
        encoder = train_utils.encode_classes(metadata_paths)
    except Exception as e:
        click.secho(f' Failed: {e}', fg='bright_red')
        return

    try:
        file_out = LABELS_FILE
        joblib.dump(encoder, file_out)
    except Exception as e:
        click.secho(f' Failed: {e}', fg='bright_red')
        return

    click.secho(' Done.', fg='bright_green')



@click.command()
@click.option('--enc', is_flag=True, help="Encode the labels and quit.")
def main(enc):

    if enc:
        encoder_export()
        return

    if not LABELS_FILE.exists():
        encoder_export()

    devices = {dev.device_type: dev.physical_device_desc for dev in device_lib.list_local_devices()}  # type: ignore

    if 'GPU' in devices.keys():
        click.secho('Tensorflow: GPU available, will use the following device for calculation:', fg='bright_green')
        click.secho('\t' + (devices.get('GPU') or ''), fg='bright_white')
    elif 'CPU' in devices.keys():
        click.secho('Tensorflow: Only CPU available, no GPU calculation possible.', fg='bright_yellow')
    else:
        click.secho('Tensorflow: No compatible device found, exiting.', fg='bright_red')
        return

    tf.compat.v1.disable_v2_behavior()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    click.secho('Loading encoded classes..', fg='bright_white')
    encoder_file = params.features_dir / 'label_encoder.joblib'
    encoder: LabelEncoder = joblib.load(encoder_file)

    click.secho('Loading training data..', fg='bright_white')
    X_train, y_train = train_utils.prepare_data(
        params.train_data_path, params.train_names_path,
        params.train_metadata_path, encoder)

    click.secho('Loading validation data..', fg='bright_white')
    X_valid, y_valid = train_utils.prepare_data(
        params.valid_data_path, params.valid_names_path,
        params.valid_metadata_path, encoder)

    # click.secho('Loading test data..', fg='bright_white')
    # X_test, y_test = train_utils.prepare_data(
    #     params.test_data_path, params.test_names_path,
    #     params.test_metadata_path, encoder)

    num_classes: int = len(encoder.classes_)
    input_shape: Tuple[int, ...] = X_train[0].shape

    model = train_utils.build_model(input_shape, num_classes)
    callbacks = train_utils.prepare_training()

    _ = train_utils.train_model(model, X_train, y_train, X_valid,
                                           y_valid, callbacks)


if __name__ == "__main__":
    main()
