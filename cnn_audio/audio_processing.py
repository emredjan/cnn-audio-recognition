import warnings
from pathlib import Path

import click
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np

from cnn_audio.params import pr

warnings.filterwarnings(action='ignore', category=UserWarning)

SR = pr['nsynth']['sample_rate']


def plot_recording(
    audio_data: np.ndarray,
    sample_rate: int = SR,
    seconds: int | None = None,
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
    seconds: int,
    window_size: int,
    hop_length: int,
    calculate: list[str],
) -> dict[str, np.ndarray]:
    audio_data, sr = librosa.load(audio_file, sr=None)  # type: ignore

    if seconds:
        audio_data = audio_data[: seconds * sr]

    features = {}
    stft = None
    chroma_stft = None
    mfcc_stft = None
    mfcc = None

    if ('chroma_stft' in calculate) or ('mfcc_stft' in calculate):
        stft = np.abs(librosa.stft(audio_data, hop_length=hop_length))


    if 'chroma_stft' in calculate:
        chroma_stft = librosa.feature.chroma_stft(
            S=stft, sr=sr, n_chroma=window_size, hop_length=hop_length
        )
        features['chroma_stft'] = chroma_stft

    if 'mfcc_stft' in calculate:
        mfcc_stft = librosa.feature.mfcc(
            y=audio_data, S=stft, sr=sr, n_mfcc=window_size, hop_length=hop_length
        )
        features['mfcc_stft'] = mfcc_stft

    if 'mfcc' in calculate:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=window_size, hop_length=hop_length)
        features['mfcc'] = mfcc

    return features


def process_single_file(
    audio_file: Path,
    seconds: int,
    window_size: int,
    hop_length: int,
    calculate: list[str],
):
    try:
        features = extract_features(
            audio_file,
            seconds=seconds,
            window_size=window_size,
            hop_length=hop_length,
            calculate=calculate,
        )
    except:
        print(audio_file.stem, 'has errors')
        features = {}

    return (audio_file.stem, features)


def dump_to_file(obj, name: str, dir: Path):
    file_name = dir / (name + '.joblib')

    return joblib.dump(obj, file_name, compress=3)


def process_files(
    audio_dir: Path,
    max_files: int | None,
    seconds: int,
    window_size: int,
    hop_length: int,
    calculate: list[str],
    label: str,
):
    file_list = list(audio_dir.glob('*.wav'))

    if max_files and (max_files < len(file_list)):
        import random

        file_list = random.sample(file_list, max_files)

    num_files = len(file_list)

    frame_size = seconds * SR
    t = int(round(frame_size / hop_length, 0))

    names = np.empty((num_files,), dtype=object)

    features: dict[str, np.ndarray] = {}
    for feature in calculate:
        features[feature] = np.empty((num_files, window_size, t))

    p_label = f"Processing {label:<5} files"

    with click.progressbar(file_list, label=p_label) as bar:

        for i, f in enumerate(bar):
            name, features_single_file = process_single_file(
                f,
                seconds=seconds,
                window_size=window_size,
                hop_length=hop_length,
                calculate=calculate,
            )

            names[i] = name

            for feature in calculate:
                features[feature][i] = features_single_file[feature]

    return (names, features)
