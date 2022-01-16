from __future__ import division
import numpy as np
import math
import os
import random
import collections
import sys
import os.path

pairs_per_folder=10
n_test=50
cmin=75

SRC_FOLDER ='/home/Speaker/Voice_All_Onefolder_active_numpy25ms10ms_hamming_cmvn_samples_3s_1s_raw'
FileDest_FOLDER='/home/Speaker/Speech/verification_Utternace'

DST_File1 =FileDest_FOLDER+'/verification_sample_train'
DST_File2 =FileDest_FOLDER+'/verification_sample_test'




MAP_File=FileDest_FOLDER+'/verification_utterance_map'
Train_File=FileDest_FOLDER+'/verification_ID_train'
Test_File=FileDest_FOLDER+'/verification_ID_test'
Train_File_map=FileDest_FOLDER+'/verification_utterance_train_map'
Test_File_map=FileDest_FOLDER+'/verification_utterance_test_map'


if not os.path.exists(FileDest_FOLDER):
    os.makedirs(FileDest_FOLDER)


c=0
folder_dict = {}
dirs=os.listdir(SRC_FOLDER)
id=[]
for file in dirs:


    file_name=SRC_FOLDER+'/'+file
    ID=file_name.split('/')[-1]
    ID=ID.split('_')[1]
    ID=ID.split('~')[0]
    id.append(ID)

    if ID in dict.keys(folder_dict):
        folder_dict[ID].append(file)
    else:
        folder_dict[ID]=[file]

    c+=1
    if c%100==0:
        print(c)


U=set(id)
print(len(U))
U=list(U)
U=sorted(U)
classnamestoids=dict(zip(U,range(len(U))))

file1 = open(MAP_File, 'w')
for d in sorted(classnamestoids):
    file1.write(str(d) + ' ' + str(classnamestoids[d]) + '\n')

file1.close()


c=0
Train_0=[]
Test_0=[]
for d in folder_dict:
    k=folder_dict[d]
    if len (k)==6:
        flag = 0
        for test in folder_dict[d]:
            path, dirs, files = os.walk(SRC_FOLDER+'/'+test).next()
            file_count = len(files)
            if file_count<cmin:
                flag=1
                break
        if flag==0:
            Test_0.append(d)
        else:
            Train_0.append(d)
    else:
        Train_0.append(d)
    c+=1
    print(c)



Test_id=np.random.choice(Test_0,size=n_test,replace=False)
Train_id=Train_0[:]

Train_id=Train_id+(list(set(Test_0)-set(Test_id)))



Train_id=sorted(Train_id)
Test_id=sorted(Test_id)

file1 = open(Train_File,'w')
file2 = open(Test_File,'w')

for d in Train_id:
    file1.write(str(d) + '\n')


for d in Test_id:
    file2.write(str(d) + '\n')

file1.close()
file2.close()
#########################################################################################

file1 = open(Train_File_map,'w')
file2 = open(Test_File_map,'w')

for d in Train_id:
    for t in folder_dict[d]:
        file1.write(str(t) + '\n')


for d in Test_id:
    for t in folder_dict[d]:
        file2.write(str(t) + '\n')

file1.close()
file2.close()


#########################################################################################
file1 = open(DST_File1,'w')
c=0
for t in Train_id:
    K=folder_dict[t]
    for n in range(pairs_per_folder):
        for k in K:
            _, _, files = os.walk(SRC_FOLDER+'/'+k).next()
            pg1=k+'/'+random.choice(files)

            P1=random.choice(K)
            _, _, files = os.walk(SRC_FOLDER + '/' + P1).next()
            pg2 = P1 + '/' + random.choice(files)

            _, _, files = os.walk(SRC_FOLDER + '/' + k).next()
            pi1 = k + '/' + random.choice(files)

            P2=Train_id[:]
            P2.remove(t)
            P2=random.choice(P2)
            Ki = folder_dict[P2]
            Ki=random.choice(Ki)

            _, _, files = os.walk(SRC_FOLDER + '/' + Ki).next()
            pi2 = Ki + '/' + random.choice(files)

            file1.write(str(pg1) + ' ' + str(pg2) +' '+ '0' + '\n')
            file1.write(str(pi1) + ' ' + str(pi2) + ' ' + '1' + '\n')
    c+=1
    print(c)

file1.close()
c=0
file2 = open(DST_File2,'w')
for t in Test_id:
    K=folder_dict[t]
    for n in range(pairs_per_folder):
        for k in K:
            _, _, files = os.walk(SRC_FOLDER+'/'+k).next()
            pg1=k+'/'+random.choice(files)

            P1=random.choice(K)
            _, _, files = os.walk(SRC_FOLDER + '/' + P1).next()
            pg2 = P1 + '/' + random.choice(files)

            _, _, files = os.walk(SRC_FOLDER + '/' + k).next()
            pi1 = k + '/' + random.choice(files)

            P2=Test_id[:]
            P2.remove(t)
            P2=random.choice(P2)
            Ki = folder_dict[P2]
            Ki=random.choice(Ki)

            _, _, files = os.walk(SRC_FOLDER + '/' + Ki).next()
            pi2 = Ki + '/' + random.choice(files)

            file2.write(str(pg1) + ' ' + str(pg2) +' '+ '0' + '\n')
            file2.write(str(pi1) + ' ' + str(pi2) + ' ' + '1' + '\n')

    c+=1
    print(c)

file2.close()


##########################################################################################33
