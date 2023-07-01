import warnings
from pathlib import Path

import click
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

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

    with click.progressbar(file_list, label="\tProcessing audio files") as bar:

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


def combine_columns(*args):
    return '_'.join(args)


def encode_classes(metadata_paths: list[Path], targets: list):
    df = pd.read_json(metadata_paths[0], orient='index')

    for mdp in metadata_paths[1:]:
        metadata = pd.read_json(mdp, orient='index')
        df = pd.concat([df, metadata], axis=0, join='outer')

    target: pd.Series = df[targets].astype(str).apply(lambda row: combine_columns(*row), axis=1, result_type=None)

    encoder = LabelEncoder()
    encoder.fit(target)

    return encoder


def encoder_export(out_file: Path, targets: list, metadata_paths) -> None | Exception:

    try:
        encoder = encode_classes(metadata_paths, targets)
    except Exception as e:
        return e

    try:
        joblib.dump(encoder, out_file)
    except Exception as e:
        return e



def write_tfrecord(data: dict[str, np.ndarray], labels: np.ndarray, file_name: Path, audio_features):

    arrays_to_combine = []

    for af in audio_features:
        arrays_to_combine.append(np.expand_dims(data[af], axis=-1))

    if len(audio_features) > 1:
        data_array = np.concatenate(arrays_to_combine, axis=-1)
    else:
        data_array = arrays_to_combine[0]

    data_shape = data_array[0].shape

    tf_file_name = str(file_name)
    with tf.io.TFRecordWriter(tf_file_name) as writer:

        feature = {}
        num_records = range(labels.shape[0])
        with click.progressbar(num_records, label='\tWriting TFRecord file') as bar:
            for i in bar:

                feature['input'] = tf.train.Feature(float_list=tf.train.FloatList(value=data_array[i].flatten()))  # type: ignore
                feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))  # type: ignore

                example = tf.train.Example(features=tf.train.Features(feature=feature))  # type: ignore
                writer.write(example.SerializeToString())  # type: ignore

    return data_shape
