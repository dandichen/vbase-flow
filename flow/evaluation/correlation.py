import numpy as np

__author__ = 'Dandi Chen'


def get_correlation(flow, flow_gt):
    valid_idx = np.logical_and(flow.mask, flow_gt.mask)

    corr_X = np.corrcoef(flow.val_x[valid_idx], flow_gt.val_x[valid_idx])[1, 0]
    corr_Y = np.corrcoef(flow.val_y[valid_idx], flow_gt.val_y[valid_idx])[1, 0]
    return corr_X, corr_Y
