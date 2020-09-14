import numpy as np
import cv2 as cv


class TemplateMatching(object):
    def __init__(self):
        # Template matching prepare
        self.methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        self.buff_size = 30
        self.buff = [0] * self.buff_size

    def recognize(self, img, template, method_num=5):
        img_ = img

        # 1. convert image to gray
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # 2. get tempate size
        template_width, template_height = template.shape[::-1]

        # 3. Execution method for template matching method
        method = self.methods[method_num]
        method = eval(method)

        # 4. Template matching with method
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # 5. If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + template_width,
                        top_left[1] + template_height)

        # 6. Crop query image via matching results
        img_crop = img_[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        return img_crop, np.min(res)

    def peak_detection(self, score):
        self.buff.pop(0)
        self.buff.append(score)

        print(self.buff)
        