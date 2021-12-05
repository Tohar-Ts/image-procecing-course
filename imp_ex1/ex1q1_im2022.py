"""
Tohar Tsvitman, 318192838
EX1 q1
"""

import numpy as np
import cv2 as cv
import json
import matplotlib.pyplot as plt


remove = '{"}'
path = 'bliss.png'


menu = {
    'a': 'Change image brightness',
    'b': 'Change image contrast',
    'c': 'Use threshold',
    'd': 'Gamma correction.'
}


def entropy(im1, im2):
    hist1, __ = np.histogram(im1, 256, [0,256])
    hist1_n = hist1 / sum(hist1)
    prob1 = hist1_n[np.nonzero(hist1_n)]                                 
    entropy1 = -np.sum(np.multiply(prob1, np.log2(prob1)))

    hist2, __ = np.histogram(im2, 256, [0,256])
    hist2_n = hist2 / sum(hist2)
    prob2 = hist2_n[np.nonzero(hist2_n)]  
    entropy2 = -np.sum(np.multiply(prob2, np.log2(prob2)))

    # hist3, __ = np.histogram(im3, 256, [0,256])
    # hist3_n = hist3 / sum(hist3)
    # prob3 = hist3_n[np.nonzero(hist3_n)]  
    # entropy3 = -np.sum(np.multiply(prob3, np.log2(prob3)))

    return entropy1, entropy2

def cdf(img):
    hist, _ = np.histogram(img, 256, [0,256])
    cdf = hist.cumsum()
    return (cdf - cdf.min())* 255 / cdf.max()


def hist_equalization(ruined_img):
    hist, bins = np.histogram(ruined_img, 256, [0,256])
    cdf = hist.cumsum()
    cdf_normal = (cdf - cdf.min())* 255 / cdf.max()
    return np.uint8(cdf_normal[ruined_img])


def change_brightness(img, b):
    img = img +b
    img[img > 255] = 255
    img[img < 0] = 0
    return np.uint8(img)



def change_contrast(img, c):
    img = img*c
    img[img > 255] = 255
    img[img < 0] = 0
    return np.uint8(img)


def threshold(img, t):
    t, img = cv.threshold(img, t, 255, cv.THRESH_BINARY_INV)
    return img


def gamma(img, g):
    mean = np.mean(img)
    g = np.log(g*255)/np.log(mean)
    img = np.power(img, g).clip(0,255)
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(int)



def user_menu():
    while True:
        print("choos one of the below option")
        print((json.dumps(menu, indent = 4)).replace('"', "").replace('{', "").replace('}', ""))
        user_input = input("insert your choise here:").lower()
        if user_input not in menu.keys():
            print('\n', "***** enter a valid option *****", '\n')
        else: return user_input


def ask_for_val(min_val, max_val):
    # print("insert value in range: ", range)
    while(True):
        user_in = input("insert value in range: [{} , {}] ".format(min_val, max_val))
        try:
            user_in_num = float(user_in)
        except:
            print("invalid input")
            continue
        if user_in_num < min_val or user_in_num > max_val:
            print("invalid input")
        else:
            break
    return user_in_num
        
def show_results(ruined_img, hist_correction_img,  ruined_entropy, hist_entropy, cdf_ruined, cdf_hist, hist_ruined, hist_hist, bins_ruined, bins_hist):

    axis1 = range(0, 256, 1)
    f, ax = plt.subplots(3, 2)
    f.set_figheight(9)
    f.set_figwidth(14)
    f.tight_layout(pad=3.0)

    ax[0,0].axis('off')
    ax[0,0].set_title('ruined image', fontsize=17)
    ax[0,0].imshow(ruined_img, cmap="gray") 

    ax[0,1].axis('off')
    ax[0,1].set_title('image after histogram equalization', fontsize=17)
    ax[0,1].imshow(hist_correction_img, cmap="gray")
    
    ax[1,0].hist(bins_ruined[:-1],bins_ruined, weights=hist_ruined , color='b')
    ax[1,0].set_title('histogram.\n the entropy is: %.3f' % ruined_entropy, fontsize=13)

    ax[1,1].hist(bins_hist[:-1],bins_hist, weights=hist_hist , color='b')
    ax[1,1].set_title('histogram.\n the entropy is: %.3f'% hist_entropy, fontsize=13)

    ax[2,0].plot(axis1, cdf_ruined, 'r')
    ax[2,0].set_title('Cumulative distribution function', fontsize=13)

    ax[2,1].plot(axis1, cdf_hist, 'r')
    ax[2,1].set_title('Cumulative distribution function', fontsize=13)

    plt.show()


if __name__ == '__main__':
    try:
        img = cv.imread(path, 0)
        if img is None:
            print("failed to read image")
    except:
        print("failed to read image")

    original_img = img
    
    user_in = user_menu()
    if user_in == "a":
        b_to_add = ask_for_val(-120,120)
        ruined_img = change_brightness(img, b_to_add)
        
    elif user_in == "b":
        contract = ask_for_val(.1,5)
        ruined_img = change_contrast(img,contract)
        
    elif user_in == "c":
        thresh_val = ask_for_val(20,200)
        ruined_img = threshold(img, thresh_val)
        
    elif user_in == "d":
        gamma_val = ask_for_val(0.1,5)
        ruined_img = gamma(img, gamma_val)
    
    hist_correction_img = hist_equalization(ruined_img)
    
    ruined_entropy, hist_entropy = entropy(ruined_img, hist_correction_img)

    hist_ruined, bins_ruined = np.histogram(ruined_img, 256, [0,256])
    hist_hist, bins_hist = np.histogram(hist_correction_img, 256, [0,256])

    cdf_ruined = cdf(ruined_img)
    cdf_hist = cdf(hist_correction_img)

    show_results(ruined_img, hist_correction_img,  ruined_entropy, hist_entropy, cdf_ruined, cdf_hist, hist_ruined, hist_hist, bins_ruined, bins_hist)


    
