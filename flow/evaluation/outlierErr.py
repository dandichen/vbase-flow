import numpy as np

__author__ = 'Dandi Chen'

# KITTI evaluation criteria: tau
def get_outlier_err(flow, flow_gt, tau=3, tau_per=0.05):
    valid_ind = np.logical_and(flow.mask, flow_gt.mask)
    gt_amp = (flow_gt.val_x[valid_ind]**2 + flow_gt.val_y[valid_ind]**2)**0.5

    delta_X = flow.val_x[valid_ind] - flow_gt.val_x[valid_ind]
    delta_Y = flow.val_y[valid_ind] - flow_gt.val_y[valid_ind]
    err_amp = (delta_X**2 + delta_Y**2)**0.5

    valid_mask = flow_gt.mask[valid_ind]
    # err_num = len(np.logical_and(valid_mask, err_amp > tau, err_amp/gt_amp > tau_per))
    err_num = len(np.logical_and(valid_mask, err_amp / gt_amp > tau_per))
    if len(valid_mask) != 0:
        err_ratio = err_num/len(valid_mask)
    else:
        err_ratio = float('nan')
    return err_ratio