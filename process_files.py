import warnings
from pathlib import Path

import click

from cnn_audio.params import pr
from cnn_audio import audio_processing as ap

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings(action='ignore', category=UserWarning)


@click.command()
@click.option('-m', '--max-files', default=None, type=int)
def main(max_files: int | None):

    output_dir_base: str = pr['locations']['features_base_dir']

    dir_label = 'SAMPLE' if max_files else 'FULL'
    output_dir = Path(output_dir_base.replace('||TYPE||', dir_label))

    output_dir.mkdir(exist_ok=True, parents=True)

    base_dir: str = pr['locations']['nsynth_data_dir']
    audio_dir: str = pr['locations']['nsynth_audio_dir_name']

    partitions = ['train', 'test', 'valid']
    partition_labels = pr['partitions']
    dirs = [Path(base_dir.replace('||PARTITION||', d)) / audio_dir for d in partitions]

    nsynth_max_secs = pr['nsynth']['max_seconds']
    librosa_spec_windows = pr['librosa']['spec_windows']
    librosa_hop_length = pr['librosa']['hop_length']

    calculate_features = pr['model']['features']

    for d, d_name in zip(dirs, partitions):

        d_label = partition_labels[d_name]

        names, features = ap.process_files(
            d,
            seconds=nsynth_max_secs,
            window_size=librosa_spec_windows,
            hop_length=librosa_hop_length,
            max_files=max_files,
            calculate=calculate_features,
            label=d_label,
        )

        names_file = ap.dump_to_file(names, d_name + '_name', output_dir)

        if not Path(names_file[0]).exists():
            click.secho('Error writing names', fg='bright_red')

        for feature in calculate_features:

            feature_file = ap.dump_to_file(features[feature], f'{d_name}_{feature}', output_dir)

            if not Path(feature_file[0]).exists():
                click.secho(f'Error writing {d_name}_{feature}', fg='bright_red')


if __name__ == "__main__":
    main()
