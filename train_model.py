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
from cnn_audio.setup_logger import get_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings(action='ignore', category=UserWarning)

NSYNTH_BASE_DIR: str = pr['locations']['nsynth_data_dir']
NSYNTH_METADATA_FILE_NAME: str = pr['locations']['nsynth_metadata_file_name']

@click.command()
@click.option('-s', '--sample', is_flag=True, help="Run only for the sample dataset.")
def main(sample):

    logger = get_logger(phase='TRAINING')

    features_dir_base: str = pr['locations']['features_base_dir']

    dir_label = 'SAMPLE' if sample else 'FULL'
    features_dir = Path(features_dir_base.replace('||TYPE||', dir_label))

    targets = pr['model']['targets']
    targets_affix = '-'.join(['t'] + targets)

    labels_file = features_dir / f'labels_{targets_affix}.joblib'

    run_id: str = pendulum.now().format('YYYYMMDD_HHmmss')

    devices = {dev.device_type: dev.physical_device_desc for dev in device_lib.list_local_devices()}  # type: ignore

    if 'GPU' in devices.keys():
        logger.info('Tensorflow: GPU available, will use the following device for calculation:')
        logger.info((devices.get('GPU') or ''))
    elif 'CPU' in devices.keys():
        logger.warning('Tensorflow: Only CPU available, no GPU calculation possible.')
    else:
        logger.error('Tensorflow: No compatible device found, exiting.')
        return

    logger.debug('Started loading encoded classes')
    encoder: LabelEncoder = joblib.load(labels_file)
    logger.info('Loaded encoded classes')

    partition_labels = pr['partitions']
    partitions = list(partition_labels.keys())
    partitions.remove('test')

    audio_features = pr['model']['features']
    feature_affix = '-'.join(['f'] + audio_features)

    datasets = {}
    batch_size=pr['model']['batch_size']

    for p in partitions:

        data_path = features_dir / f'{p}_{feature_affix}_{targets_affix}.tfrecord'
        names_path = features_dir / f'{p}_name.joblib'

        num_samples = joblib.load(names_path).shape[0]

        datasets[p] = tf.data.TFRecordDataset(filenames = [data_path])

        datasets[p] = datasets[p].map(mt.parse_tfrecord)

        datasets[p] = datasets[p].shuffle(num_samples).batch(batch_size)
        datasets[p] = datasets[p].prefetch(buffer_size=tf.data.AUTOTUNE)

    num_classes: int = len(encoder.classes_)

    data_shape_path = features_dir / f'data_shape_{feature_affix}.joblib'
    input_shape: tuple[int, ...] = joblib.load(data_shape_path)

    model, model_description = mt.build_model_1(num_classes, input_shape)
    logger.info(f"Using {model_description} for training")

    model_image_dir = Path(pr['locations']['model_image_dir'])
    model_image_dir.mkdir(exist_ok=True, parents=True)

    model_plot_file_name = Path(pr['locations']['model_image_dir']) / f'model_{run_id}.png'
    plot_model(model, to_file=str(model_plot_file_name), show_shapes=True)

    callbacks = mt.prepare_training(run_id)

    _ = mt.train_model(
        model=model,
        data_train=datasets['train'],
        data_valid = datasets['valid'],
        epochs=pr['model']['epochs'],
        batch_size=batch_size,
        callbacks=callbacks,
    )

    saved_model_file_name = Path(pr['locations']['saved_model_dir']) / f'model_{run_id}.keras'
    _ = mt.save_model(model, saved_model_file_name)


if __name__ == "__main__":
    main()
