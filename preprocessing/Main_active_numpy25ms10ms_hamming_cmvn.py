from __future__ import division
import numpy as np
import math
from Main_features import Cube2cmvn
import os
from processing import cmvn

SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming'
DST_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn'


if not os.path.exists(DST_FOLDER):
    os.makedirs(DST_FOLDER)

c=0
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file

            Cube = np.load(file_name)

            Cube_cmvn=Cube2cmvn(Cube,[0,1,2])
            out_file_name=DST_FOLDER+'/'+file.split('.')[0]

            np.save(out_file_name, Cube_cmvn)
            c+=1
            print(c)