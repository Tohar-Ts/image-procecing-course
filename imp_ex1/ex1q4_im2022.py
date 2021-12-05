"""
Tohar Tsvitman, 318192838
EX1 q4
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def create_rhombus(height, width, rhom_width):
    img = np.zeros((height, width))
    rhom = 1
    mid_ctr = 0
    start_at = rhom_width
    end_at = rhom_width*2 + start_at

    for i in range(start_at, end_at):
        if mid_ctr < rhom_width:
            img[i][2*rhom_width - rhom: 2*rhom_width + rhom] = 255
            rhom += 1
            mid_ctr+=1
        else:
            img[i][2*rhom_width - rhom: 2*rhom_width + rhom] = 255
            rhom -= 1
    return img

def add_noise(img):
    noisy_img = np.copy(img) 
    pix_count = int(img.size * 0.05)
    row_to_change = np.random.randint(0, img.shape[0]-1, pix_count)
    col_to_change = np.random.randint(0, img.shape[1]-1, pix_count)
    noise = np.random.normal(115, 70, size = pix_count)

    for i in range(0, pix_count):
        noisy_img[row_to_change, col_to_change] = noise[i]

    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img



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


def find_outlines(img):
    filter_x= np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    outline_x = use_filter(img, filter_x)
    outline_y = use_filter(img, filter_y)
    outline = outline_x + outline_y
    outline[outline < 0] = 0
    outline[outline > 256] = 255

    return outline

if __name__ == '__main__':

    rhombus = create_rhombus(400, 400, 100)
    outline = find_outlines(rhombus)
    
    noisy_image = add_noise(rhombus)
    noisy_outline = find_outlines(noisy_image)

    f, x = plt.subplots(2, 2)

    x[0,0].imshow(rhombus, cmap="gray")
    x[0,0].set_title("original rhombus")
    x[0,0].axis('off')

    x[0,1].imshow(outline, cmap="gray")
    x[0,1].set_title("otlines")
    x[0,1].axis('off')

    x[1,0].imshow(noisy_image, cmap="gray")
    x[1,0].set_title("noisy rhombus")
    x[1,0].axis('off')

    x[1,1].imshow(noisy_outline, cmap="gray")
    x[1,1].set_title("noisy outline")
    x[1,1].axis('off')

    plt.show()