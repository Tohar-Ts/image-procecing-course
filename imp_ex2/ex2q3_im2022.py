"""
Tohar Tsvitman, 318192838
EX1 q4
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import ex2q2_im2022 as q2
from numpy.core.fromnumeric import argmax


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
    noise = np.random.normal(190, 10, size = pix_count)

    for i in range(0, pix_count):
        noisy_img[row_to_change, col_to_change] = noise[i]

    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    
    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
        
    return accumulator, thetas, rhos


def find_peak(accumulator, thetas, rhos):
    idx = np.argmax(accumulator)
    r_i = int(idx / accumulator.shape[1])
    t_i = idx % accumulator.shape[1]
    rho = rhos[r_i]
    theta = thetas[t_i]
    accumulator[r_i,t_i] = 0
    return theta, rho

def find_rhombus_line(accumulator, thetas, rhos, rhombus_width):
    top_right, top_left, bottom_right, bottom_left = 0, 0, 0, 0
    num_of_edges = 4
    final_t = []
    final_r =[]
    x_range = []
    while num_of_edges > 0:
        t, r = find_peak(accumulator, thetas, rhos)
        if t > 0: # the line is increasing
            if r <290: # the line is negative
                if bottom_left >0:
                    continue
                else:
                    bottom_left+=1
                    x_range.append(range(rhombus_width,rhombus_width*2))

            else:
                if top_right >0:
                    continue
                else:
                    top_right+=1
                    x_range.append(range(rhombus_width*2,rhombus_width*3))
        elif t < 0: # the line is decreasing
            if r > 0: #the line is negative
                if bottom_right >0:
                    continue
                else:
                    bottom_right+=1
                    x_range.append(range(rhombus_width*2,rhombus_width*3))
            else:
                if top_left >0:
                    continue
                else:
                    top_left+=1
                    x_range.append(range(rhombus_width,rhombus_width*2))
        final_r.append(r)
        final_t.append(t)
        num_of_edges -=1
    return final_t, final_r, x_range


def create_outline_img( final_t, final_r, x_ranges):
    final_outline = np.zeros(noisy_outline.shape) 
    idx = 0
    for x_r in x_ranges:   
        m = -np.cos(final_t[idx])/np.sin(final_t[idx])
        b = final_r[idx]/np.sin(final_t[idx]) 
        for x in x_r:
            y = int(m*x + b)
            if (y <= 0) or (y >= final_outline.shape[0]):
                continue
            final_outline[x,y] = 255
        idx+=1
    return final_outline

if __name__ == '__main__':
    rhombus_width = 100
    img_size = 400
    rhombus = create_rhombus(img_size, img_size, rhombus_width)
    noisy_rhombus = add_noise(rhombus)
    noisy_outline = q2.canny_edge_detector(noisy_rhombus)

    accumulator, thetas, rhos = hough_line(noisy_outline)
    final_t, final_r, x_ranges = find_rhombus_line(accumulator, thetas, rhos, rhombus_width)
    final_outline = create_outline_img( final_t, final_r, x_ranges)
    
    f, x = plt.subplots(2, 2)

    x[0,0].imshow(noisy_rhombus, cmap="gray")
    x[0,0].set_title("rhombus with noise")
    x[0,0].axis('off')

    x[0,1].imshow(noisy_outline, cmap="gray")
    x[0,1].set_title("outlines with canny edge detector")
    x[0,1].axis('off')

    x[1,0].imshow(noisy_rhombus, cmap="gray")
    x[1,0].set_title("rhombus with noise")
    x[1,0].axis('off')

    x[1,1].imshow(final_outline, cmap="gray")
    x[1,1].set_title("outlines with hough transform")
    x[1,1].axis('off')

    plt.show()