"""
Tohar Tsvitman, 318192838
EX1 q3
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

    
def use_filter(img, filter):
    return cv.filter2D(img,-1,filter)


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

if __name__ == '__main__':
    rhombus = create_rhombus(400, 400, 100)

    filter_x= np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    outline_x = use_filter(rhombus, filter_x)
    outline_y = use_filter(outline_x, filter_y)
    outline = outline_x + outline_y

    outline[outline < 0] = 0
    outline[outline > 256] = 255

    f, x = plt.subplots(1, 2)
    
    x[0].imshow(rhombus, cmap="gray")
    x[0].set_title("original rhombus")
    x[0].axis('off')

    x[1].imshow(outline, cmap="gray")
    x[1].set_title("otlines")
    x[1].axis('off')

    plt.show()