from __future__ import division
import numpy as np
# from . import processing
# from scipy.fftpack import dct
import math
# from . import functions
from Main_features import Audio2numpy
from Main_features import Audio2numpy_MFCC
from Main_features import Audio2numpy_spectrogram
from Main_features import Audio2numpy_MFSC_hamming
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from processing import cmvn
from skimage import exposure

normalize_status = True
choice_stack=True
#
#
# A=np.ones(3,)
# B=np.ones(3)
# C=np.hamming(3)
# D=np.hamming(3)
# E=C*D
#
file_name='/home/Speaker/test_active.wav'




fs, signal = wav.read(file_name)
signal = signal.astype(float)

# Normalize signal
if normalize_status:
    signal = signal / 32767

if choice_stack:
    frame_length = 0.025
    overlap_factor = 0.010


# CubeS=Audio2numpy_spectrogram(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, fft_length=512)
#
# # A=CubeS[4000:4200,:,0]
# A=CubeS[1:400,:,0]
# A = np.transpose(A)
# A = exposure.equalize_hist(A)
#
# imgplot = plt.imshow(A,cmap="jet")
# plt.colorbar()
#
#
#

# Cube=Audio2numpy(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40, fft_length=512,low_frequency=0, high_frequency=None)
CubeH=Audio2numpy_MFSC_hamming(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40, fft_length=512,low_frequency=0, high_frequency=None)
# Cube1=Audio2numpy_MFCC(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,num_cepstral =13,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, dc_elimination=True)
# CubeS=Audio2numpy_spectrogram(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, fft_length=512)


E=cmvn(CubeH[:,:,2], variance_normalization=True)
E = np.transpose(E)

#
l=1
h=CubeH.shape[0]
pl=3000
ph=3400
# A=Cube[l:h,:,0]
AH=CubeH[l:h,:,2]
# A1=Cube1[l:h,:,0]
# AS=CubeS[l:h,:,0]

AH = np.transpose(AH)

B=np.copy(AH)

print(B.shape[0])
for i in range(B.shape[0]):
    m = np.min(B[i, :])
    n = np.max(B[i, :])
    B[i,:]=(B[i,:]-m)/(n-m)

C=np.copy(AH)
print(C.shape[1])
for i in range(C.shape[1]):
    m = np.min(C[:,i])
    n = np.max(C[:,i])
    C[:,i]=(C[:,i]-m)/(n-m)

D=np.copy(AH)
eps = 2 ** -30

print(D.shape[0])
for i in range(D.shape[0]):
    m = np.mean(D[i, :])

    D[i,:]=(D[i,:]-m)
    stdev = np.std(D[i,:])

    D[i,:] = D[i,:] / (stdev + eps)



fig = plt.figure()

plt.subplot(411)
plt.imshow(AH[:,pl:ph],cmap="jet")
# plt.colorbar()
plt.subplot(412)
plt.imshow(E[:,pl:ph],cmap="jet")
# plt.colorbar()
plt.subplot(413)
plt.imshow(D[:,pl:ph],cmap="jet")
# plt.colorbar()


plt.subplot(414)
plt.imshow(B[:,pl:ph],cmap="jet")


# plt.subplot(514)
# plt.imshow(C[:,pl:ph],cmap="jet")

plt.show()



