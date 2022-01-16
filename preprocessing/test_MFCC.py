from __future__ import division
import numpy as np

import math
from Main_features import Audio2numpy
from Main_features import Audio2numpy_MFCC
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



from skimage import exposure

normalize_status = True
choice_stack=True

file_name='/home/Speaker/test_active.wav'

fs, signal = wav.read(file_name)
signal = signal.astype(float)

# Normalize signal
if normalize_status:
    signal = signal / 32767

if choice_stack:
    frame_length = 0.025
    overlap_factor = 0.010

Cube=Audio2numpy_MFCC(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,num_cepstral =13,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, dc_elimination=True)

A=Cube[4000:4050,:,0]
#
# m=np.min(A)
# n=np.max(A)
#
# A=(A-m)/(n-m)
#
A = np.transpose(A)
A_eq = exposure.equalize_hist(A)
#
imgplot = plt.imshow(A_eq,cmap="jet")
plt.colorbar()
#
#
#
# Cube = np.transpose(Cube, (1, 0, 2))
#
#
# print Cube.shape
#
#
# # Cube = np.transpose(Cube, (2, 1, 0))
#
# print Cube.shape
#
#
# imgplot = plt.imshow(Cube)


np.save('cube',Cube)
