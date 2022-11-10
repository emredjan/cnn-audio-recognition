import warnings
from pathlib import Path

import click

from audiomidi import audio_utils, params

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings(action='ignore', category=UserWarning)


@click.command()
@click.option('-m', '--max-files', default=None, type=int)
def main(max_files):

    output_dir = params.features_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # process training files

    train_dir = params.nsynth_train_audio
    test_dir = params.nsynth_test_audio
    valid_dir = params.nsynth_valid_audio

    dirs = [train_dir, test_dir, valid_dir]
    dir_names = ['train', 'test', 'valid']

    for d, d_name in zip(dirs, dir_names):

        names, chroma_stfts, mfcc_stfts, _ = audio_utils.process_files(
            d,
            max_files=max_files,
            calc_chroma_stft=True,
            calc_mfcc_stft=True,
            calc_mfcc=False,
            label=d_name)

        names_file = audio_utils.dump_to_file(names, d_name + '_name',
                                              output_dir)

        if not Path(names_file[0]).exists():
            print('Error writing names')

        chroma_stft_file = audio_utils.dump_to_file(
            chroma_stfts, d_name + '_chroma_stft', output_dir)

        if not Path(chroma_stft_file[0]).exists():
            print('Error writing chroma_stft_train')

        mfcc_stft_file = audio_utils.dump_to_file(
            mfcc_stfts, d_name + '_mfcc_stft', output_dir)

        if not Path(mfcc_stft_file[0]).exists():
            print('Error writing mfcc_stft_train')


if __name__ == "__main__":
    main()  #pylint: disable=E1120
