import numpy as np
import matplotlib.pyplot


__author__ = 'haoyi liang'


class MotionStatus(object):
    def __init__(self, frame_size, v_point, vel, is_stop, cal_origin=np.nan):
        self.frame_size = frame_size
        self.v_point = v_point
        self.vel = vel
        self.cal_origin = cal_origin
        self.is_stop = is_stop

    def get_v_point(self):
        return self.v_point

    def get_dir(self):
        if np.isnan(self.cal_origin):
            origin = self.frame_size/2
        else:
            origin = self.cal_origin
        direction = np.arctan((self.v_point[:, 0] - float(origin[0]))/origin[0])/3.14*180
        return direction

