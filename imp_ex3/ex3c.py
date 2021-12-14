"""
Tohar Tsvitman, 318192838
EX3 q3
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import operator
from numpy import fft


def pad_and_rotate(dig, img):

    distance = tuple(map(operator.sub,img.shape , dig.shape)) #find padding size
    dig = np.pad(dig, [(distance[0]//2 , distance[0]//2 + 1), (distance[1]//2 , distance[1]//2)], constant_values = 0)
    dig_rotate = cv.rotate(dig, cv.ROTATE_90_COUNTERCLOCKWISE)

    return dig, dig_rotate

def transform_fourier(img, reverse):
    if not reverse:
        res = fft.fft2(img)
        res = fft.fftshift(res)
    else:
        res = fft.ifft2(img)
        res = fft.ifftshift(res)

    return res

if __name__ == '__main__':
    img = cv.imread("fft1.png", cv.IMREAD_GRAYSCALE)
    digit_img = cv.imread("4.png", cv.IMREAD_GRAYSCALE)

    digit_img , digit_rotate = pad_and_rotate(digit_img, img)

    tf_img = transform_fourier(img, False)
    tf_dig = transform_fourier(digit_img, False)
    tf_dig_rot = transform_fourier(digit_rotate, False)

    # multiply the image and the filter in the frequency plane
    tf_mult_dig = np.multiply(tf_img, tf_dig.conj())
    tf_mult_dig_rot = np.multiply(tf_img, tf_dig_rot.conj())

    dig_place = tf_mult_dig / np.abs(tf_mult_dig)
    dig_rot_place = tf_mult_dig_rot / np.abs(tf_mult_dig_rot)    
    
    dig_place = np.abs(transform_fourier(dig_place, True))
    dig_rot_place = np.abs(transform_fourier(dig_rot_place, True))

    y1,x1 = np.where(dig_place > 0.05)
    y2,x2 = np.where(dig_rot_place > 0.05)

    X = np.stack((x1, x2), axis=1)
    Y = np.stack((y1, y2), axis=1)

    plt.plot(X ,Y, 'go')
    plt.imshow(img, cmap='gray')
    # plt.savefig("find the digit 4 Q3.png")
    plt.show()
