import matplotlib.pyplot as plt
import numpy as np
from audiomidi import params

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

    fig.axes[0].set_xlabel(f'Sample ({sample_rate} Hz)')
    fig.axes[0].set_ylabel('Amplitude')
    plt.show()
