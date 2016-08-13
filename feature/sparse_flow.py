import cv2
import numpy as np
import flow_base

__author__ = 'Dandi Chen'

class SparseFlow(flow_base.Flow):
    def __init__(self, x_val, y_val, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, win_size=(15, 15), \
                        max_level=2, cridxia=(cv2.TERM_CRidxIA_EPS | cv2.TERM_CRidxIA_COUNT, 10, 0.03)):
        flow_base.Flow.__init__(self, x_val, y_val)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.win_size = win_size
        self.max_level = max_level
        self.cridxia = cridxia

    def compute(self, prvs, next, mask=None):
        # params for corner detection
        feature_params = dict(self.max_corners, self.quality_level, self.min_distance, self.block_size)

        # LK params
        lk_params = dict(self.win_size, self.max_level, self.cridxia)

        # find corners in first frame
        p0 = cv2.goodFeaturesToTrack(prvs, mask, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(prvs)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for idx in range(len(good_old)):
            self.val[np.int_(good_old[idx, 1]), np.int_(good_old[idx, 0]), 0] = (good_new - good_old)[idx, 0]  # flow x
            self.val[np.int_(good_old[idx, 1]), np.int_(good_old[idx, 0]), 1] = (good_new - good_old)[idx, 1]  # flow y
            self.val[np.int_(good_old[idx, 1]), np.int_(good_old[idx, 0]), 2] = 1  # mask
