"""
Tohar Tsvitman, 318192838
EX1 q3
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage

    
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


def create_rhombus(height, width, rhom_width):
    img = np.zeros((height, width))
    rhom = 1
    mid_ctr = 0
    start_at = rhom_width
    end_at = rhom_width*2 + start_at

    for i in range(start_at, end_at):
        if mid_ctr < rhom_width:
            img[i][2*rhom_width - rhom: 2*rhom_width + rhom] = 255 #from:to
            rhom += 1
            mid_ctr+=1
        else:
            img[i][2*rhom_width - rhom: 2*rhom_width + rhom] = 255
            rhom -= 1

    return img

def gs_filter(img):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.int32)
    return use_filter(img, kernel)


def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    D = np.arctan2(Iy, Ix)
    
    return (G, D)  


def round_angle(angle):
    angle = np.rad2deg(angle) % 180 
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle
    

def suppression(img, theta):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            # find neighbors pixels to visit from the gradient directions
            angle = round_angle(theta[i, j])
           
            if angle == 0:
                neighbor_1 = img[i, j - 1]
                neighbor_2 = img[i, j + 1]
                
            elif angle == 90:
                neighbor_1 = img[i - 1, j]
                neighbor_2 = img[i + 1, j]
                
            elif angle == 135:
                neighbor_1 = img[i - 1, j - 1]
                neighbor_2 = img[i + 1, j + 1]
    
            else: 
                neighbor_1 = img[i - 1, j + 1]
                neighbor_2 = img[i + 1, j - 1]
                
            if (img[i, j] >= neighbor_1) and (img[i, j] >= neighbor_2):
                Z[i, j] = img[i, j]
    return Z


def threshold(img, low, high):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
    'WEAK': np.int32(50),
    'STRONG': np.int32(255),
    }
    
    # get strong pixel indices
    strong_i, strong_j = np.where(img > high)
    
    # get weak pixel indices
    weak_i, weak_j = np.where((img >= low) & (img <= high))
    
    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < low)
    
    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)
    
    return (img, cf.get('WEAK'))


def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbor's is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                        or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                        or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def canny_edge_detector(img):
    gs_img  = gs_filter(img)
    grad_img, theta = gradient_intensity(np.copy(gs_img))
    non_max = suppression(np.copy(grad_img), theta)
    threshold_img, weak = threshold(np.copy(non_max), 150 , 220 )
    canny_img = tracking(np.copy(threshold_img), weak)
    return canny_img 

if __name__ == '__main__':
    rhombus = create_rhombus(400, 400, 100)

    outlines = canny_edge_detector(rhombus)

    f, x = plt.subplots(1, 2)
    
    x[0].imshow(rhombus, cmap="gray")
    x[0].set_title("original rhombus")
    x[0].axis('off')

    x[1].imshow(outlines, cmap="gray")
    x[1].set_title("otlines")
    x[1].axis('off')

    plt.show()