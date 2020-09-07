import numpy as np
import cv2 as cv


class KeypointDetection(object):
    def __init__(self, args, flann_index_kdtree=0):
        self.min_match_count = args.min_match_count
        self.flann_index_kdtree = flann_index_kdtree

        # Initiate SIFT detector
        self.encoder = cv.xfeatures2d.SIFT_create()

    def recognize(self, img1, img2):
        # img1 = cv.imread(img1, cv.IMREAD_GRAYSCALE)
        # img2 = cv.imread(img2, cv.IMREAD_GRAYSCALE)
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.encoder.detectAndCompute(img1, None)
        kp2, des2 = self.encoder.detectAndCompute(img2, None)

        # matching each feature vector
        index_params = dict(algorithm=self.flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
        else:
            matchesMask = []

        return matchesMask