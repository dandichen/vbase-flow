import cv2
import flow_base

__author__ = 'Dandi Chen'

class DenseFlow(flow_base.Flow):
    def __init__(self, x_val, y_val):
        flow_base.Flow.__init__(self, x_val, y_val)

    def compute(self, prvs, next,  \
                 flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,  \
                 poly_n=5, poly_sigma=1.2, flags=0):
        self.val = cv2.calcOpticalFlowFarneback(prvs, next, flow, pyr_scale, levels, winsize, iterations, \
                                           poly_n, poly_sigma, flags)

