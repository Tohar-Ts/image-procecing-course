"""
Tohar Tsvitman, 318192838
EX3 q4
"""

from matplotlib import image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def hough_circle(img):

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=30, param2=15, minRadius= int(img.shape[0]/2 -5), maxRadius= int(img.shape[0]/2 +5))
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    circles = np.round(circles[0, :]).astype("int")

    for (x, y ,r) in circles:
        # print(x, y, r)
        cv.circle(img, (x, y), r, (255,255,0), 2)

    return img


if __name__ == '__main__':
    img = cv.imread("fimage.png", cv.IMREAD_GRAYSCALE)
    circle = hough_circle(img)


    f,x = plt.subplots(1, 2)
    x[0].imshow(img, cmap = "gray")
    x[0].set_title("original image")
    x[0].axis("off")

    x[1].imshow(circle)
    x[1].set_title("marked circle")
    x[1].axis("off")
    # plt.savefig('find circle Q4.png')
    plt.show()