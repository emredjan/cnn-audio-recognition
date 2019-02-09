import warnings
from pathlib import Path

import click
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np

from audiomidi import params

warnings.filterwarnings(action='ignore', category=UserWarning)

def plot_recording(audio_data: np.ndarray,
                   sample_rate: int = params.nsynth_sr,
                   seconds: int = None,
                   figwidth: int = 20,
                   figheight: int = 2) -> None:

    fig = plt.figure(figsize=(figwidth, figheight))

    if seconds:
        plt.plot(audio_data[0:seconds * sample_rate])
    else:
        plt.plot(audio_data)

    fig.axes[0].set_title('Waveform')
    fig.axes[0].set_xlabel(f'Sample ({sample_rate} Hz)')
    fig.axes[0].set_ylabel('Amplitude')
    plt.show()


def extract_features(audio_file: Path,
                     seconds: int = params.nsynth_max_seconds,
                     window_size: int = params.librosa_spec_windows,
                     hop_length: int = params.librosa_hop_length,
                     calc_chroma_stft=True,
                     calc_mfcc_stft=True,
                     calc_mfcc=True):

    audio_data, sr = librosa.load(audio_file, sr=None)

    if seconds:
        audio_data = audio_data[:seconds * sr]

    stft = np.abs(librosa.stft(
        audio_data,
        hop_length=hop_length)) if calc_chroma_stft or calc_mfcc_stft else None
    chroma_stft = librosa.feature.chroma_stft(
        S=stft, sr=sr, n_chroma=window_size,
        hop_length=hop_length) if calc_chroma_stft else None
    mfcc_stft = librosa.feature.mfcc(
        audio_data, S=stft, sr=sr, n_mfcc=window_size,
        hop_length=hop_length) if calc_mfcc_stft else None
    mfcc = librosa.feature.mfcc(
        audio_data, sr=sr, n_mfcc=window_size,
        hop_length=hop_length) if calc_mfcc else None

    return (chroma_stft, mfcc_stft, mfcc)


def process_files(audio_dir: Path,
                  max_files: int = None,
                  calc_chroma_stft=True,
                  calc_mfcc_stft=True,
                  calc_mfcc=True,
                  label: str = None):

    file_list = list(audio_dir.glob('*.wav'))

    if max_files:
        file_list = file_list[:max_files]

    num_files = len(file_list)

    window_size = params.librosa_spec_windows
    hop_length = params.librosa_hop_length
    frame_size = params.nsynth_max_seconds * params.nsynth_sr
    t = int(round(frame_size / hop_length, 0))

    names = np.empty((num_files, ), dtype=object)
    chroma_stfts = np.empty((num_files, window_size,
                             t)) if calc_chroma_stft else None
    mfcc_stfts = np.empty((num_files, window_size,
                           t)) if calc_mfcc_stft else None
    mfccs = np.empty((num_files, window_size, t)) if calc_mfcc else None

    p_label = 'Processing ' + label + ' files'
    with click.progressbar(file_list, label=p_label) as bar:

        for i, f in enumerate(bar, 1):
            try:
                chroma_stft, mfcc_stft, mfcc = extract_features(
                    f,
                    calc_chroma_stft=calc_chroma_stft,
                    calc_mfcc_stft=calc_mfcc_stft,
                    calc_mfcc=calc_mfcc)
            except:
                print(f.stem, 'has errors')
                continue

            idx = i - 1

            names[idx] = f.stem

            if calc_chroma_stft:
                chroma_stfts[idx] = chroma_stft
            if calc_mfcc_stft:
                mfcc_stfts[idx] = mfcc_stft
            if calc_mfcc:
                mfccs[idx] = mfcc

            #if i % 500 == 0:
            #    print('Processed', i, '/', num_files, 'files')

            #if i == num_files:
            #    print('Processed all', i, 'files')

    return (names, chroma_stfts, mfcc_stfts, mfccs)


def dump_to_file(obj, name: str, dir: Path):

    file_name = dir / (name + '.joblib')

    return joblib.dump(obj, file_name, compress=3)
