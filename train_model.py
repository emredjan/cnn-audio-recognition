import os
import warnings
from pathlib import Path

import click
import joblib
import pendulum
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.python.client import device_lib
from keras.utils import plot_model

from cnn_audio.params import pr
from cnn_audio import model_training as mt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings(action='ignore', category=UserWarning)

NSYNTH_BASE_DIR: str = pr['locations']['nsynth_data_dir']
NSYNTH_METADATA_FILE_NAME: str = pr['locations']['nsynth_metadata_file_name']


def encoder_export(out_file: Path):

    partitions = pr['partitions'].keys()
    metadata_paths = [
        Path(NSYNTH_BASE_DIR.replace('||PARTITION||', p)) / NSYNTH_METADATA_FILE_NAME
        for p in partitions
    ]

    click.secho('Encoding classes...', fg='bright_white', nl=False)

    try:
        encoder = mt.encode_classes(metadata_paths)
    except Exception as e:
        click.secho(f' Failed: {e}', fg='bright_red')
        return

    try:
        joblib.dump(encoder, out_file)
    except Exception as e:
        click.secho(f' Failed: {e}', fg='bright_red')
        return

    click.secho(' Done.', fg='bright_green')


@click.command()
@click.option('-e', '--encode', is_flag=True, help="Encode the labels only and quit.")
@click.option('-s', '--sample', is_flag=True, help="Run only for the sample dataset.")
def main(encode, sample):
    features_dir_base: str = pr['locations']['features_base_dir']

    dir_label = 'SAMPLE' if sample else 'FULL'
    features_dir = Path(features_dir_base.replace('||TYPE||', dir_label))

    labels_file = features_dir / 'label_encoder.joblib'

    if encode:
        encoder_export(labels_file)
        return

    if not labels_file.exists():
        encoder_export(labels_file)

    run_id: str = pendulum.now().format('YYYYMMDD_HHmmss')

    devices = {dev.device_type: dev.physical_device_desc for dev in device_lib.list_local_devices()}  # type: ignore

    if 'GPU' in devices.keys():
        click.secho(
            'Tensorflow: GPU available, will use the following device for calculation:',
            fg='bright_green',
        )
        click.secho('\t' + (devices.get('GPU') or ''), fg='bright_white')
    elif 'CPU' in devices.keys():
        click.secho(
            'Tensorflow: Only CPU available, no GPU calculation possible.', fg='bright_yellow'
        )
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
            click.secho(e, fg='bright_red')

    click.secho('Loading encoded classes..', fg='bright_white')
    encoder: LabelEncoder = joblib.load(labels_file)

    features = pr['model']['features']

    partition_labels = pr['partitions']
    partitions = list(partition_labels.keys())
    partitions.remove('test')

    datasets = {}

    for p in partitions:
        p_label = partition_labels[p]
        metadata_path = (
            Path(NSYNTH_BASE_DIR.replace('||PARTITION||', p)) / NSYNTH_METADATA_FILE_NAME
        )

        feature_sets = {}
        for f in features:
            data_path = features_dir / f'{p}_{f}.joblib'
            names_path = features_dir / f'{p}_name.joblib'

            click.secho(f'Loading {p_label} data for {f}...', fg='bright_white', nl=False)
            dataset = mt.prepare_data(data_path, names_path, metadata_path, encoder)
            feature_sets[f] = dataset
            click.secho(' Done.', fg='bright_green')

        datasets[p] = feature_sets

    num_classes: int = len(encoder.classes_)
    input_shapes: list[tuple[int, ...]] = [datasets['train'][f][0][0].shape for f in features]

    model = mt.build_model(num_classes, input_shapes)


    model_plot_file_name = Path(pr['locations']['model_image_dir']) / f'model_{run_id}.png'
    plot_model(model, to_file=str(model_plot_file_name), show_shapes=True)

    callbacks = mt.prepare_training(run_id)

    if len(features) > 1:
        data_train = (
            [datasets['train'][f][0] for f in features],
            datasets['train'][features[0]][1],
        )
        data_valid = (
            [datasets['valid'][f][0] for f in features],
            datasets['valid'][features[0]][1],
        )
    else:
        data_train = (datasets['train'][features[0]][0], datasets['train'][features[0]][1])
        data_valid = (datasets['valid'][features[0]][0], datasets['valid'][features[0]][1])

    _ = mt.train_model(
        model,
        data_train,
        data_valid,
        epochs=pr['model']['epochs'],
        batch_size=pr['model']['batch_size'],
        callbacks=callbacks,
    )

    saved_model_file_name = Path(pr['locations']['saved_model_dir']) / f'model_{run_id}.keras'
    _ = mt.save_model(model, saved_model_file_name)


if __name__ == "__main__":
    main()
