import sys

from tqdm import trange
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from unittest import result
import cv2
from cv2 import threshold
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('plane.jpeg')
oriImage = image.copy()

lst_rois = []
lst_roi = []
refPt = []
print(image.shape)
def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            lst_roi.append(refPoint[0][1])
            lst_roi.append(refPoint[1][1])
            lst_roi.append(refPoint[0][0])
            lst_roi.append(refPoint[1][0])
            cv2.imshow("Cropped", roi)
            sel = np.average(roi,axis = (0,1))
            lst_rois.append(roi)
            # print(sel)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while True:

    i = image.copy()

    if not cropping:
        cv2.imshow("image", image)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
    cv2.waitKey(1)

# close all open windows
print(lst_roi)
cv2.destroyAllWindows()
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)
    # for i in range(lst_roi[0], lst_roi[2]):
    #     for j in range(lst_roi[1], lst_roi[3]):
    #         print(energy_map[i][j],' ')
    # print(energy_map)
    return energy_map

def crop_c(img, scale_c,lst_roi):
    r, c, _ = img.shape

    for i in trange(scale_c):
        # lst_roi[3] = lst_roi[3]+1
        # lst_roi[2] = lst_roi[2]+1
        # lst_roi[1] = lst_roi[1]+1
        # lst_roi[0] = lst_roi[0]+1
        img = carve_column(img,lst_roi)

    return img

def crop_r(img, scale_r,lst_roi):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img,  lst_roi[1]-lst_roi[0], lst_roi)
    img = np.rot90(img, 1, (0, 1))
    return img

def carve_column(img,lst_roi):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img,lst_roi)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[lst_roi[2]:lst_roi[3]])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img,lst_roi):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    # print(calc_energy(img))

    M = energy_map.copy()
    # print(M)
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1,r):
        for j in range(0,c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            if (lst_roi[0]<=i<=lst_roi[1] and lst_roi[2]<=j<=lst_roi[3]):
                M[i,j] = - 1000000000
            else: M[i, j] += min_energy
    # print (M)
    # print(b)
    return M, backtrack

def main():
    if len(sys.argv) != 4:
        print('usage: carver.py <r/c> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    in_filename = sys.argv[2]
    out_filename = sys.argv[3]

    img = imread(in_filename)

    if which_axis == 'r':
        out = crop_r(img, lst_roi[1]-lst_roi[0],lst_roi)
    elif which_axis == 'c':
        out = crop_c(img, lst_roi[3]-lst_roi[2],lst_roi)
    else:
        print('usage: carver.py <r/c> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()