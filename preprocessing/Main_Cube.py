from __future__ import division
import numpy as np
import math
from features import Audio2numpy
import scipy.io.wavfile as wav


def File2numpy(file_name, normalize_status=True, choice_stack = True):


    fs, signal = wav.read(file_name)
    signal = signal.astype(float)

    # Normalize signal
    if normalize_status:
        signal = signal / 32767

    if choice_stack:
        frame_length = 0.025
        overlap_factor = 0.010

    Cube = Audio2numpy(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40,fft_length=512, low_frequency=0, high_frequency=None)

    print Cube.shape

    feature_cube = np.transpose(Cube, (2, 1, 0))

    np.save('cube', Cube)
