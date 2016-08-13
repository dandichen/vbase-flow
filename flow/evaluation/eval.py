import os
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'

def vis_evaluation(per_mat, corr_X_mat, corr_Y_mat, err_ratio_mat, ang_err_mat, end_pt_err_mat):
    plt.figure()
    plt.plot(np.arange(len(per_mat)), per_mat)
    plt.title('overlap percentage')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(corr_X_mat)), corr_X_mat, 'r', label='flow x')
    plt.plot(np.arange(len(corr_Y_mat)), corr_Y_mat, 'b', label='flow y')
    plt.legend()
    plt.title('correlation coefficient')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(err_ratio_mat)), err_ratio_mat)
    plt.title('KITTI error ratio')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(ang_err_mat)), ang_err_mat)
    plt.title('angle error')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(end_pt_err_mat)), end_pt_err_mat)
    plt.title('endpoint error')
    plt.show()

def write_evaluation(per_mat, corr_X_mat, corr_Y_mat, err_ratio_mat, ang_err_mat, end_pt_err_mat, eval_path):
    fig1 = plt.figure()
    plt.plot(np.arange(len(per_mat)), per_mat)
    plt.title('overlap percentage')
    fig1.savefig(os.path.join(eval_path, 'overlap percentage.png'))

    fig2 = plt.figure()
    plt.plot(np.arange(len(corr_X_mat)), corr_X_mat, 'r', label='flow x')
    plt.plot(np.arange(len(corr_Y_mat)), corr_Y_mat, 'b', label='flow y')
    plt.legend()
    plt.title('correlation coefficient')
    fig2.savefig(os.path.join(eval_path, 'correlation coefficient.png'))

    fig3 = plt.figure()
    plt.plot(np.arange(len(err_ratio_mat)), err_ratio_mat)
    plt.title('KITTI error ratio')
    fig3.savefig(os.path.join(eval_path, 'KITTI error ratio.png'))

    fig4 = plt.figure()
    plt.plot(np.arange(len(ang_err_mat)), ang_err_mat)
    plt.title('angle error')
    fig4.savefig(os.path.join(eval_path, 'angle error.png'))

    fig5 = plt.figure()
    plt.plot(np.arange(len(end_pt_err_mat)), end_pt_err_mat)
    plt.title('endpoint error')
    fig5.savefig(os.path.join(eval_path, 'endpoint error.png'))