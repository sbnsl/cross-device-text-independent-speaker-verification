from __future__ import division
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from skimage import exposure
A=np.load('/home/Speaker/test_active.npy')
A=np.transpose(A,(1,0,2))
fig = plt.figure()

plt.subplot(311)
plt.imshow(A[:,:,0],cmap="jet")
plt.subplot(312)
plt.imshow(A[:,:,1],cmap="jet")
plt.subplot(313)
plt.imshow(A[:,:,2],cmap="jet")
plt.show()
