from __future__ import division
import numpy as np
import math
import os
#
SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s'
DST_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'

if not os.path.exists(DST_FOLDER):
    os.makedirs(DST_FOLDER)

c=0
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file
            Cube = np.load(file_name)



            idfolder=DST_FOLDER+'/'+root.split('/')[-1]

            if not os.path.exists(idfolder):
                os.makedirs(idfolder)

            out_file_name=idfolder+'/'+file.split('.')[0]
            Cube.tofile(out_file_name)

            c+=1
            print(c)