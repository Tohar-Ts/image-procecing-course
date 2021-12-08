"""
Tohar Tsvitman, 318192838
EX3 q1
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def use_threashold(img, threashold = 127):
    img = img.copy()
    img[img > threashold] = 255
    img[img < threashold] = 0
    return img

def fix_brightness(img, kernel_size = 40):
    """fixing and equalizing the brightness to ease the use of threshold"""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))
    img_grayscale = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img_fixed = img - img_grayscale
    return img_fixed


def region_filling(outlines, show_proccess = True):
    
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    inverte = 255 - outlines #the inverted image, because we want to take the pixels that are on = 255 = whites

    filled = np.zeros(outlines.shape, np.uint8)
    filled[0][0] = 255 # turn on the first pixel from which we starts to dilate

    while True:

        dilate_filled = cv.dilate(filled, kernel, iterations=2) # dilate twice 
        dilate_filled =cv.bitwise_and(dilate_filled, inverte) #take the cut-off between the dilated image and the inverted image

        if np.array_equal(filled, dilate_filled):
            break

        filled = dilate_filled

        if show_proccess:
            cv.imshow("f", filled)
            cv.waitKey(1)
    
    if show_proccess:
        cv.imshow("f", np.bitwise_not(filled))
        cv.waitKey(1) 

    return np.bitwise_not(filled)
    
if __name__ == '__main__':
    img = cv.imread("rice.jpg", cv.IMREAD_GRAYSCALE)
    # img= plt.imread("rice.jpg")
    img_fixed_brithness = fix_brightness(img, 60)

    img_th = use_threashold(img_fixed_brithness, 50)
    
    #clean noises using open:
    kernel = np.ones((10,10),np.uint8)
    img_clear = cv.morphologyEx(img_th, cv.MORPH_OPEN, kernel)

    #find outlines using erousion
    kernel = np.ones((5,5),np.uint8)
    img_erode = cv.erode(img_clear, kernel, iterations=1)
    outlines =img_clear - img_erode

    filled = region_filling(outlines, show_proccess=False)
   
    f, x = plt.subplots(1, 3)
    x[0].imshow(img, cmap = "gray")
    x[0].set_title("original image")
    x[0].axis("off")

    x[1].imshow(outlines, cmap = "gray")
    x[1].set_title("image outlines")
    x[1].axis("off")

    x[2].imshow(filled, cmap = "gray")
    x[2].set_title("image filled")
    x[2].axis("off")

    plt.show()