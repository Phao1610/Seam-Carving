import cv2
from seam import SeamCarving

import os
import numpy as np
import cv2
import argparse
from numba import jit
from scipy import ndimage as ndi
from CreateMask import getMaskObject



filename_input = '11.png'
filename_output = 'image_result.png'
filename_mask = 'mask.png'
img = cv2.imread(filename_input)
print(img.shape)
# h,w = img.shape[:2]
if (img.shape[0]>1080 and img.shape[1]<1920):
	img = cv2.resize(img, ( img.shape[1],600))
else:
	if (img.shape[0]<1080 and img.shape[1]>1920):
		img = cv2.resize(img, (1200, img.shape[0]))
	else:
		if (img.shape[0]>1080 and img.shape[1]>1920):
			img = cv2.resize(img,(1200,600))


mask_1 = getMaskObject(img)
draw1 = mask_1[0]
mask1 = mask_1[1]

mask_2 = getMaskObject(img)
draw2 = mask_2[0]
mask2 = mask_2[1]

# Init SeamCarving Object
img = cv2.imread(filename_input)

if ((img.shape[0]!=mask1.shape[0]) or (img.shape[1]!=mask1.shape[1])):
	mask1 = cv2.resize(mask1,(img.shape[1],img.shape[0]))
	mask2 = cv2.resize(mask2,(img.shape[1],img.shape[0]))

cv2.imwrite('mask0.png', mask1)
cv2.imwrite('mask1.png',mask2)
# if (img.shape[0]>1080 and img.shape[1]<1920):
# 	img = cv2.resize(img, (800, img.shape[1]))
# else:
# 	if (img.shape[0]<1080 and img.shape[1]>1920):
# 		img = cv2.resize(img, (img.shape[0], 1500))
# 	else:
# 		if (img.shape[0]>1080 and img.shape[1]>1920):
# 			img = cv2.resize(img,(1200,600))
print(img.shape)
print(mask1.shape)
print(mask2.shape)

SC = SeamCarving(img)

result = SC.remove_object(mask1,mask2)

# Visualize process by process.gif
SC.visual_process('process_11.gif')