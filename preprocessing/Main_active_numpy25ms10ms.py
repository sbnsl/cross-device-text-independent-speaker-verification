from __future__ import division
import numpy as np
import math
from Main_features import Audio2numpy
import scipy.io.wavfile as wav
import os
from skimage import exposure

normalize_status = True
choice_stack=True

SRC_FOLDER ='/home/Speaker/Voice_all_onefolder_active'
DST_FOLDER='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms'


if choice_stack:
    frame_length = 0.025
    overlap_factor = 0.010


if not os.path.exists(DST_FOLDER):
    os.makedirs(DST_FOLDER)

c=0
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file

            fs, signal = wav.read(file_name)
            signal = signal.astype(float)

            if normalize_status:
                signal = signal / 32767



            Cube = Audio2numpy(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40,
                               fft_length=512, low_frequency=0, high_frequency=None)
            out_file_name=DST_FOLDER+'/'+file.split('.')[0]

            np.save(out_file_name, Cube)

            c+=1
            print(c)