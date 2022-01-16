from __future__ import division
import numpy as np
from processing import stack_frames
import math
# from . import functions
from feature import mfe
from feature import mfe_hamming
from feature import mfcc
from feature import spectrogram
from feature import extract_derivative_feature
from processing import cmvn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Audio2numpy(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):

    M,E=mfe(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)

    Cube=extract_derivative_feature(M)

    return Cube


def Audio2numpy_spectrogram(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01, fft_length=512):

    M,E=spectrogram(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01, fft_length=512)

    Cube=extract_derivative_feature(M)

    return Cube


def Audio2numpy_MFCC(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01,num_cepstral =13,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, dc_elimination=True):

    M=mfcc(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01,num_cepstral =13,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, dc_elimination=True)

    Cube=extract_derivative_feature(M)

    return Cube


def Audio2numpy_MFSC_hamming(signal, sampling_frequency, frame_length=0.020, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None):

    M,E=mfe_hamming(signal, sampling_frequency, frame_length=0.025, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)

    Cube=extract_derivative_feature(M)

    return Cube


def Cube2cmvn(Cube, vec):


    for i in range (Cube.shape[2]):
        if i in vec:
            Cube[:,:,i]=cmvn(Cube[:,:,i], variance_normalization=True)

    return Cube



def PlotCube(Cube):

    if Cube.ndim==2:

        a=1
    else:
        s=Cube.shape[2]









