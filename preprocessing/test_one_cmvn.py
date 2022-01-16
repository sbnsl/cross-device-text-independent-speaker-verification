from __future__ import division
import numpy as np
import math
from Main_features import Cube2cmvn
import os
from processing import cmvn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_name='/home/Speaker/test_active.npy'
Cube = np.load(file_name)
# Cube = Cube.astype(float)

Cube_cmvn=Cube2cmvn(Cube,[0,1,2])
out_file_name='testcmvn'
np.save(out_file_name, Cube_cmvn)
E=np.load('/home/Speaker/testcmvn.npy')

pl=3000
ph=3400

E = np.transpose(E,(1,0,2))
fig = plt.figure()

plt.subplot(311)
plt.imshow(E[:,pl:ph,0],cmap="jet")

plt.subplot(312)
plt.imshow(E[:,pl:ph,1],cmap="jet")

plt.subplot(313)
plt.imshow(E[:,pl:ph,2],cmap="jet")

plt.show()


