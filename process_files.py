import warnings
from pathlib import Path

import click

from audiomidi import audio_utils, params

warnings.filterwarnings(action='ignore', category=UserWarning)

@click.command()
@click.option('-m', '--max-files', default=None, type=int)
def main(max_files):

    output_dir = params.features_dir

    # process training files

    train_dir = params.nsynth_train_audio

    names, chroma_stfts, _, _ = audio_utils.process_files(
        train_dir,
        max_files=max_files,
        calc_chroma_stft=True,
        calc_mfcc_stft=False,
        calc_mfcc=False)

    names_train = audio_utils.dump_to_file(names, 'train_names', output_dir)

    if not Path(names_train[0]).exists():
        print('Error writing names')

    chroma_stft_train = audio_utils.dump_to_file(
        chroma_stfts, 'train_chroma_stft', output_dir)

    if not Path(chroma_stft_train[0]).exists():
        print('Error writing chroma_stft_train')


if __name__ == "__main__":
    main() #pylint: disable=E1120
