import cv2
import math
import numpy as np
from numpy import linalg as la
import scipy.spatial.distance as sci_dis

from keypoint_matching.flow import Flow

__author__ = 'Dandi Chen'

def read_gt(gt_path):
    val = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = val.shape

    flow_gt = Flow(width, height)
    # KITTI definition
    flow_gt.val_x = (np.float_(val[:, :, 2]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_gt.val_y = (np.float_(val[:, :, 1]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_gt.mask = np.array(val[:, :, 0], dtype=bool)

def convert_gt_mask_box(flow_mask, bbox):
    new_flow_mask = np.zeros_like(flow_mask)
    new_flow_mask[int(bbox.top_left_y):int(bbox.bottom_right_y), int(bbox.top_left_x):int(bbox.bottom_right_x)] = \
        flow_mask[int(bbox.top_left_y):int(bbox.bottom_right_y), int(bbox.top_left_x):int(bbox.bottom_right_x)]
    return new_flow_mask

def normalize_len(old_arr, start, end):
    old_min = np.amin(old_arr)
    old_range = np.amax(old_arr) - old_min

    new_range = end - start
    new_arr = [(n - old_min) / old_range * new_range + start for n in old_arr]

    return new_arr

def normalize_mat(old_arr1, old_arr2):  # normalize to center
    height1, width1 = old_arr1.shape
    height2, width2 = old_arr2.shape

    min_height = np.min(height1, height2)
    min_width = np.min(width1, width2)

    new_arr1 = old_arr1[((height1 - min_height)/2):((height1 + min_height)/2),
               ((width1 - min_width)/2):((width1 + min_width)/2)]

    new_arr2 = old_arr2[((height2 - min_height) / 2):((height2 + min_height) / 2),
               ((width2 - min_width) / 2):((width2 + min_width) / 2)]

    return new_arr1, new_arr2, min_width, min_height

def convert(old_x, old_y, old_flow_X, old_flow_Y, flow_mask_gt):
    height, width = flow_mask_gt.shape
    flow_new = Flow(width, height)

    for i in range(len(old_x)):
        flow_new.mask[old_y[i], old_x[i]] = True
        flow_new.val_x[old_y[i], old_x[i]] = old_flow_X[i]
        flow_new.val_y[old_y[i], old_x[i]] = old_flow_Y[i]
    return flow_new

def normalize_coordinate_box(x, y, bbox):
    pos_x = (x - bbox.top_left_x) / bbox.width
    pos_y = (y - bbox.top_left_y) / bbox.height
    return pos_x, pos_y

def check_range(x, y, width, height):
    if x >= 0 and x < width and y >= 0 and y < height:
        return True
    else:
        return False

def get_vector_sim(flow, flow_gt, width, height, win_size=3, ang_thld=15, amp_thld=15):
    y = np.where(flow.mask == True)[0]
    x = np.where(flow.mask == True)[1]

    sim_mask = np.zeros_like(flow_gt.mask, dtype=bool)

    for j in y:
        for i in x:
            u = [flow.val_x[j, i], flow.val_y[j, i]]
            ave_ang = []
            ave_amp = []
            for n in range((win_size - 1)/2, (win_size + 1)/2 + 2):
                for m in range((win_size - 1)/2, (win_size + 1)/2 + 2):
                    if check_range(i - 2 + m, j - 2 + n, width, height) and flow_gt.mask[j - 2 + n, i - 2 + m] == True:
                        v = [flow_gt.val_x[j - 2 + n, i - 2 + m], flow_gt.val_y[j - 2 + n, i - 2 + m]]
                        if la.norm(u) * la.norm(v) != 0:
                            ang = math.acos(np.dot(u, v) / (la.norm(u) * la.norm(v)))
                            amp = sci_dis.euclidean(u, v)
                            ave_ang.append(ang)
                            ave_amp.append(amp)
                        else:
                            continue
                    else:
                        continue
            if len(ave_ang) != 0 and len(ave_amp) != 0 and np.mean(ave_ang) < ang_thld and np.mean(ave_amp) < amp_thld:
                sim_mask[j, i] = True
    return sim_mask

# overlap between flow and keypoint correspondence
def mask2vec_mask(flow_mask, matches, kp, matchesMask):
    mask_vector = np.zeros_like(matchesMask, dtype=bool)
    for match_idx in range(len(matches)):
        idx = matches[match_idx].queryIdx
        (x, y) = kp[idx].pt
        if flow_mask[int(y), int(x)] == True:
            mask_vector[match_idx] = matchesMask[match_idx]
    return mask_vector













