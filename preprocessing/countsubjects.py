from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys


SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
L=os.listdir(SRC_FOLDER)

folders=[]
for f in L:
    i=f.split('_')[1]+f.split('_')[2]
    folders.append(i)


print(len(set(folders)))
c=0
id=[]
for root, dirs, files in os.walk(os.path.join(SRC_FOLDER)):
        for file in files:
            file_name=root+'/'+file

            ID=file_name.split('/')[-1]
            ID=ID.split('_')[1]
            ID=ID.split('~')[0]
            id.append(ID)
            c+=1
            if c%10000==0:
                print(c)


U=set(id)
print(len(U))
U=list(U)
U=sorted(U)
classnamestoids=dict(zip(U,range(len(U))))


print(classnamestoids)

