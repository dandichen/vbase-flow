import math
import numpy as np
from numpy import linalg as la

__author__ = 'Dandi Chen'


def get_angular_err(flow, flow_gt):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    valid_idx = np.logical_and(flow.mask, flow_gt.mask)

    dot_prod = 1.0 + np.dot(flow.val_x[valid_idx], flow_gt.val_x[valid_idx]) + \
               np.dot(flow.val_y[valid_idx], flow_gt.val_y[valid_idx])
    norm_X = (1.0 + la.norm(flow.val_x[valid_idx]) ** 2 + la.norm(flow_gt.val_x[valid_idx]) ** 2) ** 0.5
    norm_Y = (1.0 + la.norm(flow_gt.val_y[valid_idx]) ** 2 + la.norm(flow_gt.val_y[valid_idx]) ** 2) ** 0.5

    if norm_X * norm_Y != 0:
        val = dot_prod/(norm_X * norm_Y)

        while (val > 1 or val < -1):
            print 'angle out of range'
            if val > 1:
                val -= 2
            else:
                val += 2

        ang_err_rad = math.acos(val)  # radius
        ang_err_deg = math.degrees(ang_err_rad)  # degree

    else:
        ang_err_rad = float('nan')
        ang_err_deg = float('nan')

    return ang_err_rad, ang_err_deg




