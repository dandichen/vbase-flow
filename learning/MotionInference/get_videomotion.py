import cv2
import matplotlib.pyplot as plt
from hid_inference import HidVarInf
import numpy as np
from scipy import signal


class GetVideoMotion(object):
    def __init__(self, sparse_corner_num=100, moving_angle_thresh=[30, 10],
                 stop_thresh=1, ite_num=5, is_sparse=0, temporal_down_sample=5, demo=False):
        self.sparse_corner_num = sparse_corner_num
        self.moving_angle_thresh = moving_angle_thresh
        self.stop_thresh = stop_thresh
        self.ite_num = ite_num
        self.demo = demo
        self.is_sparse = is_sparse
        self.temporal_down_sample = temporal_down_sample

    def get_motion(self, video_path, roi, resize_ratio):
        infer = HidVarInf(sparse_corner_num=self.sparse_corner_num,
                          moving_angle_thresh=self.moving_angle_thresh,
                          stop_thresh=self.stop_thresh, ite_num=self.ite_num, is_sparse=self.is_sparse, demo=self.demo)
        video_cap = cv2.VideoCapture(video_path)
        frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
        ret, cur_frame = video_cap.read()
        cur_frame = cur_frame[roi[0][0]:roi[0][1], roi[1][0]: roi[1][1]]
        height, width, _ = cur_frame.shape
        final_size = int(width*resize_ratio),  int(height*resize_ratio)

        v_point_hist = []
        vel_hist = []
        is_stop_hist = []
        cost_hist = []
        dir_hist = []
        frame_num = 0

        while True:
            print frame_num
            ret, pre_frame = video_cap.read()
            if not ret:
                break
            frame_num += 1
            pre_frame = pre_frame[roi[0][0]:roi[0][1], roi[1][0]: roi[1][1]]
            pre_frame = cv2.resize(pre_frame, final_size)
            ret, cur_frame = video_cap.read()
            if not ret:
                break
            frame_num += 1

            cur_frame = cur_frame[roi[0][0]:roi[0][1], roi[1][0]: roi[1][1]]
            cur_frame = cv2.resize(cur_frame, final_size)
            v_point, vel, is_stop, cost, direction = infer.get_motion(pre_frame, cur_frame)
            v_point_hist.append(v_point)
            vel_hist.append(vel)
            is_stop_hist.append(is_stop)
            cost_hist.append(cost)
            dir_hist.append(direction)
            for i in range(self.temporal_down_sample - 2):
                ret, cur_frame = video_cap.read()
                if not ret:
                    break
                frame_num += 1
            if not ret:
                break

        return frame_rate, v_point_hist, vel_hist, is_stop_hist, cost_hist, dir_hist







