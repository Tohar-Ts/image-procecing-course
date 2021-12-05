"""
Tohar Tsvitman, 318192838
EX2 q1
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def use_filter(img, kernel):
    kernel_size = len(kernel)
    row = img.shape[0] - kernel_size + 1
    col = img.shape[1] - kernel_size + 1
    res_img = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            crr_pixels = img[i:i+kernel_size, j:j+kernel_size] # take filter-size pixels from img to work on
            res_img[i, j] = np.abs(sum(sum(crr_pixels * kernel))) #convolution and posting
    return res_img


def create_board(size = 150):
    img = np.zeros((size, size))
    sqr_size = size//3 
    sqr_start = [(0, sqr_size) , (sqr_size,0), (sqr_size, sqr_size*2), (sqr_size*2, sqr_size)]

    for sq in sqr_start:
        start_i, start_j = sq
        end_i = start_i+sqr_size 
        end_j = start_j+sqr_size 

        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                if i == size or j == size:
                    continue
                img[i][j] = 255

    return img

if __name__ == "__main__":
    img = create_board()
    
    filter_log = np.array([[-1, 2, -1],
                          [2, -4, 2],
                          [-1, 2, -1]])
    
    img_log = use_filter(img, filter_log)
    dots_index = np.nonzero(img_log)
    
    implot = plt.imshow(img, cmap="gray")
    plt.scatter(dots_index[0], dots_index[1], c='r', s=40)
    plt.show()


