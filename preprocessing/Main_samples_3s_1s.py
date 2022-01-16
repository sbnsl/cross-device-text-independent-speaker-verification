from __future__ import division
import numpy as np
import math
import os

SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn'
DST_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s'

if not os.path.exists(DST_FOLDER):
    os.makedirs(DST_FOLDER)

nframespersample=300
samplestride=100
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        f=0
        for file in files:
            file_name=root+'/'+file
            Cube = np.load(file_name)

            c=0
            while c<Cube.shape[0]-nframespersample+1:
                Sample=Cube[c:c+nframespersample,:,:]

                idfolder=DST_FOLDER+'/'+file.split('.')[0]

                if not os.path.exists(idfolder):
                    os.makedirs(idfolder)

                out_file_name=idfolder+'/'+file.split('.')[0]+'_'+str(c)
                np.save(out_file_name, Sample)
                c+=samplestride
                print ('file='+str(f)+ ' sample='+str(c))
            f+=1
            # print(f)