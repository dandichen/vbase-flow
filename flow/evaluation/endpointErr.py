import numpy as np

__author__ = 'Dandi Chen'


def get_endpoint_err(flow, flow_gt):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    valid_idx = np.logical_and(flow.mask, flow_gt.mask)

    delta_X = flow.val_x[valid_idx] - flow_gt.val_x[valid_idx]
    delta_Y = flow.val_y[valid_idx] - flow_gt.val_y[valid_idx]

    err_amp = (delta_X**2 + delta_Y**2)**0.5
    return np.mean(err_amp)