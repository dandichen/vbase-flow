import cv2
import numpy as np

from matcher import Matcher as Matcher

import evaluation.form as form

__author__ = 'Dandi Chen'

class BruteForceMatcher(Matcher):
    def __init__(self, key_pt_pair):
        Matcher.__init__(self, key_pt_pair)

    def get_matcher(self):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(self.key_pt_pair.des1, self.key_pt_pair.des2)
        matches = sorted(matches, key=lambda x: x.distance)
        self.matches = matches
        self.match_len = len(matches)

    def get_good_matcher(self, threshold=0.7):     # Lowe's ratio test for normalized matches distance
        good = []
        dis_norm = self._normalize_dis()
        for idx in range(self.match_len):
            if self.matches[idx].distance < threshold * self.matches[-1].distance:
                self.matches[idx].distance = dis_norm[idx]
                good.append(self.matches[idx])
        self.matches = good
        self.match_len = len(good)

    def _get_euclidean_vec_dis(self):
        dis_vec = []
        for match_idx in range(self.match_len):
            img1_idx = self.matches[match_idx].queryIdx
            img2_idx = self.matches[match_idx].trainIdx

            dis = self.key_pt_pair.get_euclidean_dis(img1_idx, img2_idx)
            dis_vec.append(dis)
        dis_vec_norm = form.normalize_len(dis_vec, 0, 1)
        return dis_vec_norm

    def _get_weight_dis(self, weight=0.5):
        wgt_dis = []
        dis_pos = self._get_euclidean_vec_dis()
        for mat_idx in range(self.match_len):
            wgt_dis.append((1 - weight) * dis_pos[mat_idx] + weight * self.matches[mat_idx].distance)
        return wgt_dis

    def get_wgt_dis_matcher(self, weight=0.5):
        wgt_dis = self._get_weight_dis(weight)
        wgt_dis_matches = self.matches
        for idx in range(self.match_len):
            wgt_dis_matches[idx].distance = wgt_dis[idx]
        self.matches = sorted(wgt_dis_matches, key=lambda x: x.distance)
        self.match_len = len(self.matches)

    def get_homography(self, src=None, min_match_count=10, ransacReprojThreshold=3.0):
        if self.match_len > min_match_count:
            src_pts = np.float32([self.key_pt_pair.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.key_pt_pair.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
            self.matchesMask = np.array(mask.ravel(), dtype=bool)

            if src != None:
                dst = cv2.perspectiveTransform(src, M)
            else:
                dst = None
        else:
            print "Not enough matches are found - %d/%d" % (self.match_len, min_match_count)
            dst = None
        return dst

    def _normalize_dis(self, start=0, end=1):
        dis = []
        for match in self.matches:
            dis.append(match.distance)
        dis_norm = form.normalize_len(dis, start, end)
        return dis_norm










