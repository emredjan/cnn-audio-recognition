import warnings
from pathlib import Path

import click
import joblib
import pandas as pd

from cnn_audio.params import pr
from cnn_audio import preprocessing as ap
from cnn_audio.setup_logger import get_logger

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings(action='ignore', category=UserWarning)


@click.command()
@click.option('-m', '--max-files', default=None, type=int)
@click.option('-j', '--export-joblib', is_flag=True)
@click.option('-t', '--export-tfrecord', is_flag=True)
@click.option('-e', '--export-encoder', is_flag=True)
def main(max_files: int | None, export_joblib: bool, export_tfrecord: bool, export_encoder: bool):

    logger = get_logger(phase='PREPROCESSING')

    output_dir_base: str = pr['locations']['features_base_dir']

    dir_label = 'SAMPLE' if max_files else 'FULL'
    output_dir = Path(output_dir_base.replace('||TYPE||', dir_label))

    output_dir.mkdir(exist_ok=True, parents=True)

    base_dir: str = pr['locations']['nsynth_data_dir']
    audio_dir: str = pr['locations']['nsynth_audio_dir_name']
    nsynth_metadata_file_name: str = pr['locations']['nsynth_metadata_file_name']

    partition_labels = pr['partitions']
    partitions = partition_labels.keys()
    dirs = [Path(base_dir.replace('||PARTITION||', d)) / audio_dir for d in partitions]

    nsynth_max_secs = pr['nsynth']['max_seconds']
    librosa_spec_windows = pr['librosa']['spec_windows']
    librosa_hop_length = pr['librosa']['hop_length']

    calculate_features = pr['model']['features']
    targets = pr['model']['targets']

    labels_file_stem = 'label_encoder'
    labels_file = None

    if export_joblib:

        for d, d_name in zip(dirs, partitions):

            d_label = partition_labels[d_name]

            logger.debug(f"Started processing audio files for {d_label} data")
            names, features = ap.process_files(
                d,
                seconds=nsynth_max_secs,
                window_size=librosa_spec_windows,
                hop_length=librosa_hop_length,
                max_files=max_files,
                calculate=calculate_features,
            )
            logger.info(f"Processed audio files for {d_label} data")


            logger.debug(f"Started writing names file for {d_label}")
            names_file = ap.dump_to_file(names, d_name + '_name', output_dir)
            logger.info(f"Names file for {d_label} written as: {Path(names_file[0])}")

            if not Path(names_file[0]).exists():
                logger.error(f"Failed writing names file for {d_label}")

            for feature in calculate_features:

                logger.debug(f"Started writing {feature} feature file for {d_label} data")
                feature_file = ap.dump_to_file(features[feature], f'{d_name}_{feature}', output_dir)
                logger.info(f"{feature} feature file for {d_label} data written as: {feature_file[0]}")

                if not Path(feature_file[0]).exists():
                    logger.error(f"Failed writing {feature} feature file for {d_label} data")

    if export_encoder:

        metadata_paths = [
            Path(base_dir.replace('||PARTITION||', p)) / nsynth_metadata_file_name
            for p in partitions
        ]

        logger.debug("Started encoding target classes")
        encoder = ap.encode_classes(metadata_paths, targets)
        logger.info("Encoded target classes")

        logger.debug("Started writing encoded target classes to file")
        labels_file_ = ap.dump_to_file(encoder, labels_file_stem, output_dir)
        labels_file = Path(labels_file_[0])
        logger.info(f"Written encoded target classes to file: {labels_file}")


    if export_tfrecord:

        for partition in partitions:

            logger.debug(f"Started processing TFRecord file for {partition_labels[partition]} data")

            tf_record_file = output_dir / f'{partition}.tfrecord'

            data_files = [output_dir / f"{partition}_{feature}.joblib" for feature in calculate_features]
            names_file = output_dir / f"{partition}_name.joblib"
            metadata_path = Path(base_dir.replace('||PARTITION||', partition)) / nsynth_metadata_file_name

            if not labels_file:
                labels_file = output_dir / f"{labels_file_stem}.joblib"

            logger.debug(f"Started loading data for {partition_labels[partition]}")
            data = {feature: joblib.load(data_path) for feature, data_path in zip(calculate_features, data_files)}
            names = joblib.load(names_file)
            encoder = joblib.load(labels_file)
            metadata = pd.read_json(metadata_path, orient='index')

            df = pd.DataFrame({}, index=names).merge(
                    metadata, how='left', left_index=True, right_index=True
                )

            label: pd.Series = df[targets].astype(str).apply(lambda row: ap.combine_columns(*row), axis=1, result_type=None)
            label_enc: np.ndarray = encoder.transform(label)  # type: ignore

            logger.debug(f"Loaded data for {partition_labels[partition]}")


            logger.debug(f"Started writing TFRecord file for {partition_labels[partition]} data")
            data_shape = ap.write_tfrecord(data, label_enc, tf_record_file, calculate_features)
            _ = ap.dump_to_file(data_shape, 'data_shape', output_dir)
            logger.info(f"Written TFRecord file for {partition_labels[partition]} data as: {tf_record_file}")

            if not tf_record_file.exists():
                logger.error("Failed writing tfrecord file")


if __name__ == "__main__":
    main()
