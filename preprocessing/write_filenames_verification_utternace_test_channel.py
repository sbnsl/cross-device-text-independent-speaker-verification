from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys
import os.path

pairs_per_folder=100
# n_test=50
# cmin=75

Channel1='_Ac_'
Channel2='_Ac_'


SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
FileDest_FOLDER='/home/Speaker/Speech/verification_Utternace_Phone2Phone'

# DST_File1 =FileDest_FOLDER+'/verification_sample_train'
# DST_File2 =FileDest_FOLDER+'/verification_sample_test'
# MAP_File=FileDest_FOLDER+'/verification_utterance_map'
# Train_File=FileDest_FOLDER+'/verification_ID_train'
# Test_File=FileDest_FOLDER+'/verification_ID_test'
# Train_File_map=FileDest_FOLDER+'/verification_utterance_train_map'
Test_File_map=FileDest_FOLDER+'/verification_utterance_test_map'
DST_File_Test =FileDest_FOLDER+'/verification_sample_test_utterance'
DST_File_Test_u =FileDest_FOLDER+'/verification_utterance_test_utterance'


file1 = open(Test_File_map, 'r')
filenames_Test = []

for line in file1:
    filename= line[:-1].split(' ')
    filenames_Test.append(filename[0])

file1.close()

c=0
folder_dict = {}
Test_id=[]
for file in filenames_Test:

    file_name=SRC_FOLDER+'/'+file
    ID=file_name.split('/')[-1]
    ID=ID.split('_')[1]
    ID=ID.split('~')[0]
    Test_id.append(ID)

    if ID in dict.keys(folder_dict):
        folder_dict[ID].append(file)
    else:
        folder_dict[ID]=[file]

    c+=1
    if c%100==0:
        print(c)

Test_id=set(Test_id)
Test_id=list(Test_id)
Test_id=sorted(Test_id)
#########################################################################################



file1 = open(DST_File_Test_u,'w')
file2 = open(DST_File_Test,'w')
c=0

for t in Test_id:
    K=folder_dict[t]
    for P1 in K:
        if Channel1 in P1:
            P2gg=K[:]
            P2gg.remove(P1)
            P2g=''
            while not(Channel2 in P2g):
                P2g=random.choice(P2gg)

            P2ii=Test_id[:]
            P2ii.remove(t)
            P2i=random.choice(P2ii)

            Ki=folder_dict[P2i]
            P2i=''
            while not(Channel2 in P2i):
                P2i=random.choice(Ki)

            file1.write(str(P1) + ' ' + str(P2g) + ' ' + '0' + '\n')
            file1.write(str(P1) + ' ' + str(P2i) + ' ' + '1' + '\n')


            for n in range(pairs_per_folder):
                _, _, files = os.walk(SRC_FOLDER+'/'+P1).next()
                pg1=P1+'/'+random.choice(files)

                _, _, files = os.walk(SRC_FOLDER + '/' + P2g).next()
                pg2 = P2g + '/' + random.choice(files)

                file2.write(str(pg1) + ' ' + str(pg2) +' '+ '0' +' '+ str(c)+ '\n')


            c+=1
            print(c)


            for n in range(pairs_per_folder):
                _, _, files = os.walk(SRC_FOLDER + '/' + P1).next()
                pi1 = P1 + '/' + random.choice(files)

                _, _, files = os.walk(SRC_FOLDER + '/' + P2i).next()
                pi2 = P2i + '/' + random.choice(files)

                file2.write(str(pi1) + ' ' + str(pi2) + ' ' + '1' + ' ' + str(c) + '\n')

            c += 1
            print(c)

file1.close()
file2.close()