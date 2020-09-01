	#!/usr/bin/env python3
# encoding: utf-8

import cv2 as cv
import numpy as np


class Matching(object):
    def __init__(self, args):
        self.tm = TemplateMatching()
        self.kpd = KeypointDetection(args)

    def baro_matching(self, img, templates):
        img_crops = []
        results = []
        detected_positions = []
        for i, template in enumerate(templates):
            _, img_crop, _, detected_position = self.tm.recognize(img, template)
            img_crops.append(img_crop)

            kps = self.kpd.recognize(img_crop, template)
            results.append(np.sum(kps))

            detected_positions.append(detected_position)

        return results, img_crops, detected_positions


class TemplateMatching(object):
    def __init__(self):
        # Template matching prepare
        self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        self.template_threshold = 0.08

    def recognize(self, img, template, method_num=5):
        img_ = img
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # template = cv.Canny(template, 0, 200)
        # img = cv.Canny(img, 0, 200)

        template_width, template_height = template.shape[::-1]

        detection_flag = False
        # Execution method for template matching method
        method = self.methods[method_num]
        method = eval(method)

        # Template matching with method
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + template_width,
                        top_left[1] + template_height)

        # Template score compare with score
        if np.min(res) < self.template_threshold:
            # cv2.rectangle(img_, top_left, bottom_right, (255, 0, 0), 3)
            detection_flag = True

        # Crop query image via matching results
        crop_img = img_[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        return img_, crop_img, detection_flag, (top_left, bottom_right)


class KeypointDetection(object):
    def __init__(self, args, flann_index_kdtree=0):
        self.min_match_count = args.min_match_count
        self.flann_index_kdtree = flann_index_kdtree

        # Initiate SIFT detector
        self.encoder = cv.xfeatures2d.SIFT_create()

    def forward(self):
        pass

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

class Endpoint(object):
    def __init__(self, guard=4):
        self.kpd_buffer = [0] * 11
        self.threshold = 10

        self.interval_guard = guard
        self.interval_counter = 0

    def detect_peak(self, new_buffer):
        return_peak_value = 0
        self.kpd_buffer.pop(0)
        self.kpd_buffer.append(new_buffer[0])

        idx = 0
        mask = np.array(self.kpd_buffer) > self.threshold

        is_end = mask[-1] or mask[-2] or mask[-3]

        if not is_end and (self.interval_counter > self.interval_guard):
            idx = np.argmax(self.kpd_buffer)
            is_peak = True and mask[idx]
            self.interval_counter = 0
            return_peak_value = self.kpd_buffer[idx]

        else:
            is_peak = False
            self.interval_counter += 1

        return is_peak, return_peak_value, idx