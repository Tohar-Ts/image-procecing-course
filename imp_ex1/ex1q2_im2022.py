"""
Tohar Tsvitman, 318192838
EX1 q2
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = 'bliss.png'

if __name__ == '__main__':
    try:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            print("failed to read image")
    except:
        print("failed to read image")
    
    red = img[:,:,2]
    blue = img[:,:,0]
    green = img[:,:,1]

    sky_pix = 0
    clouds_pix = 0
    grass_pix = 0

    w = img.shape[0]
    h = img.shape[1]

    blue_range = range(110,251)
    green_range = range(90,165)

    for i in range(w):
        for j in range(h):
            r = red[i][j]
            g = green[i][j]
            b = blue[i][j]

            if r > 180 and g > 180 and b > 180:
                clouds_pix+=1
            elif b > g:
                sky_pix+=1
            else:
                grass_pix+=1

print("  number of grass pixels: {} \n  number of clouds pixels: {} \n  number of sky pixels: {} \n".format(grass_pix, clouds_pix, sky_pix) )