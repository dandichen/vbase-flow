import cv2
import numpy as np
import operator

from matcher import MatcherList

__author__ = 'Dandi Chen'


class BruteForceMatcherList(MatcherList):
    def __init__(self, key_pt_pair_list, mList=None, distance=None, trainIdx=None, queryIdx=None,
                 imgIdx=None, mask=None, length=0):
        MatcherList.__init__(self, key_pt_pair_list, mList, distance, trainIdx, queryIdx, imgIdx, mask, length)

    def init_val(self, matches):
        MatcherList.init_val(self, matches)

    def set_val(self, idx):
        MatcherList.set_val(self, idx)

    def get_matcher(self):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(self.key_pt_pair_list.list1.des, self.key_pt_pair_list.list2.des)
        # matches = sorted(matches, key=lambda x: x.distance)
        self.init_val(matches)

    def get_good_matcher(self, threshold=0.7):
        idx = np.where(self.distance < threshold * np.amax(self.distance))
        self.set_val(idx)

    def get_wgt_dis_matcher(self, weight=0.5):
        self.key_pt_pair_list.get_euclidean_dis()
        self.distance = (1 - weight) * self.distance + weight * self.key_pt_pair_list.distance

    def get_homography(self, src=None, min_match_count=10, threshold=3.0):
        if self.length > min_match_count:
            src_pts = np.array([self.key_pt_pair_list.list1.pt_x, self.key_pt_pair_list.list1.pt_y]).reshape(-1, 1, 2)
            dst_pts = np.array([self.key_pt_pair_list.list2.pt_x, self.key_pt_pair_list.list2.pt_y]).reshape(-1, 1, 2)

            Mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
            self.mask = np.array(mask.ravel(), dtype=bool)

            if src != None:
                dst = cv2.perspectiveTransform(src, Mat)
            else:
                dst = None
        else:
            print "Not enough matches are found - %d/%d" % (self.length, min_match_count)
            dst = None
        return dst



















