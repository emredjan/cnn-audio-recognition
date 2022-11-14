from typing import Union
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

import click
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np

from audiomidi import params

warnings.filterwarnings(action='ignore', category=UserWarning)


def plot_recording(
    audio_data: np.ndarray,
    sample_rate: int = params.nsynth_sr,
    seconds: Union[int, None] = None,
    figwidth: int = 20,
    figheight: int = 2,
) -> None:

    fig = plt.figure(figsize=(figwidth, figheight))

    if seconds:
        plt.plot(audio_data[0 : seconds * sample_rate])
    else:
        plt.plot(audio_data)

    fig.axes[0].set_title('Waveform')
    fig.axes[0].set_xlabel(f'Sample ({sample_rate} Hz)')
    fig.axes[0].set_ylabel('Amplitude')
    plt.show()


def extract_features(
    audio_file: Path,
    seconds: int = params.nsynth_max_seconds,
    window_size: int = params.librosa_spec_windows,
    hop_length: int = params.librosa_hop_length,
    calc_chroma_stft=True,
    calc_mfcc_stft=True,
    calc_mfcc=True,
):

    audio_data, sr = librosa.load(audio_file, sr=None)  # type: ignore

    if seconds:
        audio_data = audio_data[: seconds * sr]

    stft = (
        np.abs(librosa.stft(audio_data, hop_length=hop_length))
        if calc_chroma_stft or calc_mfcc_stft
        else None
    )
    chroma_stft = (
        librosa.feature.chroma_stft(
            S=stft, sr=sr, n_chroma=window_size, hop_length=hop_length
        )
        if calc_chroma_stft
        else None
    )
    mfcc_stft = (
        librosa.feature.mfcc(
            audio_data, S=stft, sr=sr, n_mfcc=window_size, hop_length=hop_length
        )
        if calc_mfcc_stft
        else None
    )
    mfcc = (
        librosa.feature.mfcc(
            audio_data, sr=sr, n_mfcc=window_size, hop_length=hop_length
        )
        if calc_mfcc
        else None
    )

    return (chroma_stft, mfcc_stft, mfcc)


# def process_filesX(
#     audio_dir: Path,
#     max_files: int = None,
#     calc_chroma_stft=True,
#     calc_mfcc_stft=True,
#     calc_mfcc=True,
#     label: str = '',
#     max_workers: int = 12
# ):

#     file_list = list(audio_dir.glob('*.wav'))

#     if max_files:
#         file_list = file_list[:max_files]

#     num_files = len(file_list)

#     window_size = params.librosa_spec_windows
#     hop_length = params.librosa_hop_length
#     frame_size = params.nsynth_max_seconds * params.nsynth_sr
#     t = int(round(frame_size / hop_length, 0))

#     names = np.empty((num_files,), dtype=object)
#     chroma_stfts = np.empty((num_files, window_size, t)) if calc_chroma_stft else None
#     mfcc_stfts = np.empty((num_files, window_size, t)) if calc_mfcc_stft else None
#     mfccs = np.empty((num_files, window_size, t)) if calc_mfcc else None

#     click.secho('Processing ' + label + ' files', fg='bright_white')

#     process_file_ = partial(
#         process_single_file,
#         calc_chroma_stft=calc_chroma_stft,
#         calc_mfcc_stft=calc_mfcc_stft,
#         calc_mfcc=calc_mfcc,
#     )

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(tqdm(executor.map(process_file_, file_list), total=num_files))
#         #results = executor.map(process_file_, file_list)

#     for i, r in enumerate(results):
#         names[i] = r[0]
#         if calc_chroma_stft:
#             chroma_stfts[i] = r[1]
#         if calc_mfcc_stft:
#             mfcc_stfts[i] = r[2]
#         if calc_mfcc:
#             mfccs[i] = r[3]

#     return (names, chroma_stfts, mfcc_stfts, mfccs)


def process_single_file(
    audio_file: Path, calc_chroma_stft=True, calc_mfcc_stft=True, calc_mfcc=True
):

    try:
        chroma_stft, mfcc_stft, mfcc = extract_features(
            audio_file,
            calc_chroma_stft=calc_chroma_stft,
            calc_mfcc_stft=calc_mfcc_stft,
            calc_mfcc=calc_mfcc,
        )
    except:
        print(audio_file.stem, 'has errors')
        chroma_stft, mfcc_stft, mfcc = None, None, None

    return (audio_file.stem, chroma_stft, mfcc_stft, mfcc)


def dump_to_file(obj, name: str, dir: Path):

    file_name = dir / (name + '.joblib')

    return joblib.dump(obj, file_name, compress=3)


def process_files(
    audio_dir: Path,
    max_files: Union[int, None] = None,
    calc_chroma_stft=True,
    calc_mfcc_stft=True,
    calc_mfcc=True,
    label: str = '',
):

    file_list = list(audio_dir.glob('*.wav'))

    if max_files and (max_files < len(file_list)):
        import random
        file_list = random.sample(file_list, max_files)

    num_files = len(file_list)

    window_size = params.librosa_spec_windows
    hop_length = params.librosa_hop_length
    frame_size = params.nsynth_max_seconds * params.nsynth_sr
    t = int(round(frame_size / hop_length, 0))

    names = np.empty((num_files,), dtype=object)
    chroma_stfts = np.empty((num_files, window_size, t)) if calc_chroma_stft else None
    mfcc_stfts = np.empty((num_files, window_size, t)) if calc_mfcc_stft else None
    mfccs = np.empty((num_files, window_size, t)) if calc_mfcc else None

    p_label = 'Processing ' + label + ' files'
    with click.progressbar(file_list, label=p_label) as bar:

        for i, f in enumerate(bar):

            name, chroma_stft, mfcc_stft, mfcc = process_single_file(
                    f,
                    calc_chroma_stft=calc_chroma_stft,
                    calc_mfcc_stft=calc_mfcc_stft,
                    calc_mfcc=calc_mfcc,
                )

            names[i] = name

            if calc_chroma_stft:
                chroma_stfts[i] = chroma_stft  # type: ignore
            if calc_mfcc_stft:
                mfcc_stfts[i] = mfcc_stft  # type: ignore
            if calc_mfcc:
                mfccs[i] = mfcc  # type: ignore

    return (names, chroma_stfts, mfcc_stfts, mfccs)
