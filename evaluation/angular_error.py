import math
import numpy as np
from numpy import linalg as la

import correlation as corr

__author__ = 'Dandi Chen'

def normalize(flow, max_val, min_val):
    return (flow - min_val)/(np.float_(max_val - min_val))

def normalize_flow(flow_X, flow_Y, flow_X_gt, flow_Y_gt):
    flow_X_nr = normalize(flow_X, np.max(flow_X), np.min(flow_X))
    flow_Y_nr = normalize(flow_Y, np.max(flow_Y), np.min(flow_Y))
    flow_X_gt_nr = normalize(flow_X_gt, np.max(flow_X_gt), np.min(flow_X_gt))
    flow_Y_gt_nr = normalize(flow_Y_gt, np.max(flow_Y_gt), np.min(flow_Y_gt))

    return flow_X_nr, flow_Y_nr, flow_X_gt_nr, flow_Y_gt_nr

def get_angular_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, width, height):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    flow_X_re, flow_Y_re, flow_X_gt_re, flow_Y_gt_re = corr.reshape_flow(flow_X, flow_Y, flow_X_gt, flow_Y_gt, width, height)
    flow_X_nr, flow_Y_nr, flow_X_gt_nr, flow_Y_gt_nr = normalize_flow(flow_X_re, flow_Y_re, flow_X_gt_re, flow_Y_gt_re)

    dot_prod = 1.0 + np.dot(flow_X_nr, flow_X_gt_nr) + np.dot(flow_Y_nr, flow_Y_gt_nr)
    norm_X = (1.0 + la.norm(flow_X_nr) ** 2 + la.norm(flow_X_gt_nr) ** 2) ** 0.5
    norm_Y = (1.0 + la.norm(flow_Y_nr) ** 2 + la.norm(flow_Y_gt_nr) ** 2) ** 0.5

    val = dot_prod/(norm_X * norm_Y)

    while(val > 1 or val < -1):
        print 'angle out of range'
        if val > 1:
            val -= 2
        else:
            val += 2

    ang_err_rad = math.acos(val)                # radius
    ang_err_deg = math.degrees(ang_err_rad)     # degree

    return ang_err_rad, ang_err_deg