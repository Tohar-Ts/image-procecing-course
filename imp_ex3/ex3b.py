"""
Tohar Tsvitman, 318192838
EX3 q2
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import fft
WHITE = 255
BLACK = 0

def create_masks(shape):
    
    row, col = shape
    row, col  = row//2, col//2
    mask_size = 9

    #create the mask with the white background
    mask1 = np.full(shape, 1, np.uint16)
    mask1[:, col-mask_size : col+mask_size] = 0

    mask2 = np.full(shape, 1, np.uint16)
    mask2[row - mask_size : row + mask_size, :] = 0
    
    mask6 = np.full(shape, 1, np.uint16)
    mask6[row - mask_size : row + mask_size, col- mask_size : col + mask_size] = 0

    mask7 = np.full(shape, 1, np.uint16)
    mask7[:, col-mask_size : col+mask_size] = 0
    mask7[row - mask_size : row + mask_size, :] = 0

    mask3 = cv.bitwise_not(mask1)
    
    mask4 = cv.bitwise_not(mask2)
   
    mask5 = cv.bitwise_not(mask6)
    
    mask8 = cv.bitwise_not(mask7) 

    return [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]


def fft_with_mask(img, masks):
    img_c = np.copy(img)
    fft_img = fft.fft2(img_c)
    fft_img_shift = fft.fftshift(fft_img)
    images_fft = []

    for m in masks:

        img_mask = fft_img_shift * m

        final = np.fft.ifftshift(img_mask)

        fft_final = np.fft.ifft2(final)

        images_fft.append(np.uint16(np.abs(fft_final)))

    return images_fft


if __name__ == '__main__':
    img = cv.imread("fimage_.png", cv.IMREAD_GRAYSCALE)
    
    f, x = plt.subplots(4, 4)
    f.subplots_adjust(hspace=.4)

    masks = create_masks(img.shape)
    fft_images = fft_with_mask(img, masks)

    #ploting
    idx = 1
    for m,f in zip(masks, fft_images):

        plt.subplot(4,4,idx)
        plt.imshow(m, cmap = "gray")
        plt.axis("off")
        plt.title("mask {}".format(idx//2+1))

        plt.subplot(4,4,idx+1)
        plt.imshow(f, cmap = "gray")
        plt.axis("off")
        plt.title("result {}".format(idx//2+1))

        idx +=2
    # plt.savefig('masks and results Q2.png')
    plt.show()