"""
Tohar Tsvitman, 318192838
EX3 q2
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def create_masks(shape):
    
    row, col  = int(shape/2), int(shape/2)
    urow, drow = row+10, row-10
    ucol, dcol = col+10, col-10

    masks = []
    mask1 = np.ones(shape, np.uint16)
    mask1[:,col-10:col+10] = 0

    mask2 = np.ones(shape, np.uint16)
    mask2[row-10:row+10,:] = 0
    
    mask3 = cv.bitwise_not(mask1)
    
    mask4 = cv.bitwise_not(mask2)
   
    mask5 = np.zeros(shape, np.uint16)
    mask5[row-10:row+10,col-10:col+10] = 255
    
    mask6 = cv.bitwise_not(mask5)
    
    mask7 = np.copy(mask1)
    mask7[row-10:row+10,:] = 0
    
    mask8 = cv.bitwise_not(mask7) 
    
    
if __name__ == '__main__':
    img = plt.imread("fimage_.png")
    plt.imshow(img)
    plt.show()