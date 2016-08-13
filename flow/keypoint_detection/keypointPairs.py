import cv2
import numpy as np

import scipy.spatial.distance as sci_dis

import evaluation.form as form

__author__ = 'Dandi Chen'

def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2

class KeypointPair(object):
    def __init__(self, bbox1, bbox2, kp1=None, kp2=None, des1=None, des2=None):
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.kp1 = kp1
        self.kp2 = kp2
        self.des1 = des1
        self.des2 = des2
        self.neighbor_mat = None


    def get_euclidean_dis(self, idx1, idx2):
        (x1, y1) = self.kp1[idx1].pt
        (x2, y2) = self.kp2[idx2].pt
        pos_x1, pos_y1 = form.normalize_coordinate_box(x1, y1, self.bbox1)
        pos_x2, pos_y2 = form.normalize_coordinate_box(x2, y2, self.bbox2)

        pt1 = np.array([pos_x1, pos_y1])
        pt2 = np.array([pos_x2, pos_y2])
        dis = sci_dis.euclidean(pt1, pt2)
        return dis

    # check whether keypoint pair are neighbors
    def get_neighbor(self, threshould=0.9):
        self.neighbor_mat = np.zeros((len(kp1), len(kp2)), dtype=bool)
        for idx2 in range(len(self.kp2)):
            for idx1 in range(len(self.kp1)):
                dis = self.get_euclidean_dis(idx1, idx2)
                if dis <= threshould:
                    self.neighbor_mat[idx1][idx2] = True

    def vis_pt_pairs(self, img1, img2):
        shown_img1 = img1
        shown_img2 = img2

        if self.bbox1.top_left_x == 0 and self.bbox1.top_left_y == 0 \
                and self.bbox2.top_left_x == 0 and self.bbox2.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.kp1, shown_img1, color=(0, 255, 0), flags=0)
            cv2.imshow('image1', shown_img1)
            cv2.waitKey(0)

            shown_img2 = cv2.drawKeypoints(img2, self.kp2, shown_img2, color=(0, 255, 0), flags=0)
            cv2.imshow('image2', shown_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if self.bbox1.top_left_x != 0 or self.bbox1.top_left_y != 0:
                for kp_idx1 in self.kp1:
                    (x, y) = kp_idx1.pt
                    x += self.bbox1.top_left_x
                    y += self.bbox1.top_left_y
                    cv2.circle(shown_img1, (int(x), int(y)), 3, color=(0, 255, 0))
                cv2.imshow('image1', shown_img1)
                cv2.waitKey(0)

            if self.bbox2.top_left_x != 0 or self.bbox2.top_left_y != 0:
                for kp_idx2 in self.kp2:
                    (x, y) = kp_idx2.pt
                    x += self.bbox2.top_left_x
                    y += self.bbox2.top_left_y
                    cv2.circle(shown_img2, (int(x), int(y)), 3, color=(0, 255, 0))
                cv2.imshow('image2', shown_img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def write_pt_pairs(self, img1, img2, kp_path1, kp_path2):
        shown_img1 = img1
        shown_img2 = img2

        if self.bbox1.top_left_x == 0 and self.bbox1.top_left_y == 0 \
                and self.bbox2.top_left_x == 0 and self.bbox2.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.kp1, shown_img1, color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path1, shown_img1)

            shown_img2 = cv2.drawKeypoints(img2, self.kp2, shown_img2, color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path2, shown_img2)
        else:
            if self.bbox1.top_left_x != 0 or self.bbox1.top_left_y != 0:
                for kp_idx1 in self.kp1:
                    (x, y) = kp_idx1.pt
                    x += self.bbox1.top_left_x
                    y += self.bbox1.top_left_y
                    cv2.circle(shown_img1, (int(x), int(y)), 2, color=(0, 255, 0))
                cv2.imwrite(kp_path1, shown_img1)

            if self.bbox2.top_left_x != 0 or self.bbox2.top_left_y != 0:
                for kp_idx2 in self.kp2:
                    (x, y) = kp_idx2.pt
                    x += self.bbox2.top_left_x
                    y += self.bbox2.top_left_y
                    cv2.circle(shown_img2, (int(x), int(y)), 2, color=(0, 255, 0))
                cv2.imwrite(kp_path2, shown_img2)

