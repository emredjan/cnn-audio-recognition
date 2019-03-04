from pathlib import Path
from typing import Tuple

import click
import joblib
from sklearn.preprocessing import LabelEncoder

from audiomidi import params, train_utils


def encoder_export():

    metadata_paths = [
        params.train_metadata_path, params.valid_metadata_path,
        params.test_metadata_path
    ]

    click.secho('Encoding classes..', fg='bright_white')
    encoder = train_utils.encode_classes(metadata_paths)

    file_out: Path = params.features_dir / 'label_encoder.joblib'

    joblib.dump(encoder, file_out)



@click.command()
def main():

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

    click.secho('Loading test data..', fg='bright_white')
    X_test, y_test = train_utils.prepare_data(
        params.test_data_path, params.test_names_path,
        params.test_metadata_path, encoder)

    num_classes: int = len(encoder.classes_)
    input_shape: Tuple[int, ...] = X_train[0].shape

    model = train_utils.build_model(input_shape, num_classes)
    callbacks = train_utils.prepare_training()

    fitted_model = train_utils.train_model(model, X_train, y_train, X_valid,
                                           y_valid, callbacks)


if __name__ == "__main__":
    main()
