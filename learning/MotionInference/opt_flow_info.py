import numpy as np
import matplotlib.pyplot


__author__ = 'haoyi liang'


class OptFlowInfo(object):
    def __init__(self, position, flow_vector, confidence, is_static, depth):
        self.position = position
        self.flow_vector = flow_vector
        self.confidence = confidence
        self.is_static = is_static
        self.depth = depth

    def get_v_point(self):
        return self.v_point

    def get_dir(self):
        if np.isnan(self.cal_origin):
            origin = self.frame_size/2
        else:
            origin = self.cal_origin
        direction = np.arctan((self.v_point[:, 0] - float(origin[0]))/origin[0])/3.14*180
        return direction
