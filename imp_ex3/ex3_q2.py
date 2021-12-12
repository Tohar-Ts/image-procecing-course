"""
Tohar Tsvitman, 318192838
EX3 q2
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def create_masks(shape):
    
    row, col = shape
    row, col  = int(row/2), int(col/2)

    mask1 = np.ones(shape, np.uint16)
    mask1[:,col-10:col+10] = 0

    mask2 = np.ones(shape, np.uint16)
    mask2[row-10:row+10,:] = 0
    
    mask3 = cv.bitwise_not(mask1)
    
    mask4 = cv.bitwise_not(mask2)
   
    mask5 = np.zeros(shape, np.uint16)
    mask5[row-10:row+10, col-10:col+10] = 255
    
    mask6 = cv.bitwise_not(mask5)
    
    mask7 = np.copy(mask1)
    mask7[row-10:row+10,:] = 0
    
    mask8 = cv.bitwise_not(mask7) 

    return [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]


def fft(img, masks):
    fft_img = np.fft.fft2(img)
    images_fft = []

    for i, m in enumerate(masks):
        img_mask = np.multiply(fft_img, m)
        shifted = np.fft.ifftshift(img_mask)
        fft_final = np.fft.ifft2(shifted)
        images_fft.append(np.uint16(np.abs(fft_final)))

    return images_fft


if __name__ == '__main__':
    img = cv.imread("fimage_.png", cv.IMREAD_GRAYSCALE)
    print(img.shape)
    
    f, x = plt.subplots(4, 2)
    masks = create_masks(img.shape)
    m = 0
    for r in range(2):
        for c in range(4):
            x[c,r].imshow(masks[m], cmap = "gray")
            x[c,r].axis("off")
            m+=1
    plt.show()

    fft_images = fft(img, masks)

    f, x = plt.subplots(4, 2)
    f_img = 0
    for r in range(2):
        for c in range(4):
            x[c,r].imshow(fft_images[f_img], cmap = "gray")
            x[c,r].axis("off")
            f_img+=1
    plt.show()