import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def image_stitch(images, lowe_ratio=0.75, max_Threshold=4.0,match_status=False):

    #detect the features and keypoints from SIFT
    (imageB, imageA) = images
    (KeypointsA, features_of_A) = Detect_Feature_And_KeyPoints(imageA)
    (KeypointsB, features_of_B) = Detect_Feature_And_KeyPoints(imageB)

    #got the valid matched points
    Values = matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

    if Values is None:
        return None

    #to get perspective of image using computed homography
    (matches, Homography, status) = Values
    result_image = getwarp_perspective(imageA,imageB,Homography)
    result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # check to see if the keypoint matches should be visualized
    if match_status:
        vis = draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)

        return (result_image, vis)

    return result_image

def getwarp_perspective(imageA,imageB,Homography):
    val = imageA.shape[1] + imageB.shape[1]
    result_image = cv.warpPerspective(imageA, Homography, (val , imageA.shape[0]))

    return result_image

def Detect_Feature_And_KeyPoints(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # detect and extract features from the image
    descriptors = cv.xfeatures2d.SIFT_create()
    (Keypoints, features) = descriptors.detectAndCompute(image, None)

    Keypoints = np.float32([i.pt for i in Keypoints])
    return (Keypoints, features)

def get_Allpossible_Match(featuresA,featuresB):

    # compute the all matches using euclidean distance and opencv provide
    #DescriptorMatcher_create() function for that
    match_instance = cv.DescriptorMatcher_create("BruteForce")
    All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

    return All_Matches

def All_validmatches(AllMatches,lowe_ratio):
    #to get all valid matches according to lowe concept..
    valid_matches = []

    for val in AllMatches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            valid_matches.append((val[0].trainIdx, val[0].queryIdx))

    return valid_matches

def Compute_Homography(pointsA,pointsB,max_Threshold):
    #to compute homography using points in both images

    (H, status) = cv.findHomography(pointsA, pointsB, cv.RANSAC, max_Threshold)
    return (H,status)

def matchKeypoints( KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

    AllMatches = get_Allpossible_Match(featuresA,featuresB)
    valid_matches = All_validmatches(AllMatches,lowe_ratio)

    if len(valid_matches) > 4:
        # construct the two sets of points
        pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
        pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

        (Homograpgy, status) = Compute_Homography(pointsA, pointsB, max_Threshold)

        return (valid_matches, Homograpgy, status)
    else:
        return None

def get_image_dimension(image):
    (h,w) = image.shape[:2]
    return (h,w)

def get_points(imageA,imageB):

    (hA, wA) = get_image_dimension(imageA)
    (hB, wB) = get_image_dimension(imageB)
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    return vis


def draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches, status):

    (hA,wA) = get_image_dimension(imageA)
    vis = get_points(imageA,imageB)

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
            ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
            cv.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis


if __name__ == "__main__":

    print("\n\nOption 1: View from a window (4 images, best results)\n\nOption 2: Catan game (3 images, good results)\n\nOption 3: Teddy bear (2 images, bad results)")
    op = input ("\nchoose option number:")

    if op == "1":
        img1 = cv.imread("images/1.1.jpeg")
        img2 = cv.imread("images/1.2.jpeg")
        img3 = cv.imread("images/1.3.jpeg")
        img4 = cv.imread("images/1.4.jpeg")

        img1 = cv.resize(img1, (0, 0), None, 0.7, 0.7)
        img2 = cv.resize(img2, (0, 0), None, 0.7, 0.7)
        img3 = cv.resize(img3, (0, 0), None, 0.7, 0.7)
        img4 = cv.resize(img4, (0, 0), None, 0.7, 0.7)

        (result, matched_points) = image_stitch([img2, img1], match_status=True)
        (result, matched_points) = image_stitch([img3, result], match_status=True)
        (result, matched_points) = image_stitch([img4, result], match_status=True)

        result = result[:,:2000]
        # cv.imwrite("panorama 1.jpg", result)

        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
        img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)
        img4 = cv.cvtColor(img4, cv.COLOR_RGB2BGR)

        plt.subplot(2,2,1)
        plt.imshow(result)
        plt.axis("off")
        plt.title("panorama image ")

        plt.subplot(2,4,5)
        plt.imshow(img4)
        plt.axis("off")
        plt.title("1st image")

        plt.subplot(2,4,6)
        plt.imshow(img3)
        plt.axis("off")
        plt.title("2nd image")

        plt.subplot(2,4,7)
        plt.imshow(img2)
        plt.axis("off")
        plt.title("3rd image ")

        plt.subplot(2,4,8)
        plt.imshow(img1)
        plt.axis("off")
        plt.title("4th image ")

        # plt.savefig("create panorama 1.jpg")
        plt.show()

    elif op == "2":
        img1 = cv.imread("images/2.1.jpeg")
        img2 = cv.imread("images/2.2.jpeg")
        img3 = cv.imread("images/2.3.jpeg")

        (result, matched_points) = image_stitch([img2, img1], match_status=True)
        (result, matched_points) = image_stitch([img3, result], match_status=True)

        result = result[:,:2800]
        # cv.imwrite("panorama 2.jpg", result)

        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
        img3 = cv.cvtColor(img3, cv.COLOR_RGB2BGR)

        plt.subplot(2,2,1)
        plt.imshow(result)
        plt.axis("off")
        plt.title("panorama image")

        plt.subplot(2,3,4)
        plt.imshow(img3)
        plt.axis("off")
        plt.title("1st image")

        plt.subplot(2,3,5)
        plt.imshow(img2)
        plt.axis("off")
        plt.title("2nd image")

        plt.subplot(2,3,6)
        plt.imshow(img1)
        plt.axis("off")
        plt.title("3rd image")

        # plt.savefig("create panorama 2.jpg")
        plt.show()

    elif op == "3":
        img1 = cv.imread("images/3.1.jpeg")
        img2 = cv.imread("images/3.2.jpeg")

        (result, matched_points) = image_stitch([img1, img2], match_status=True)

        # result = result[:,:2800]
        # cv.imwrite("panorama 3.jpg", result)

        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)

        plt.subplot(2,2,1)
        plt.imshow(result)
        plt.axis("off")
        plt.title("panorama image")

        plt.subplot(2,2,3)
        plt.imshow(img2)
        plt.axis("off")
        plt.title("1st image")

        plt.subplot(2,2,4)
        plt.imshow(img1)
        plt.axis("off")
        plt.title("2nd image")

        # plt.savefig("create panorama 3.jpg")
        plt.show()
