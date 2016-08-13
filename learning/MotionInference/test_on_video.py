from get_videomotion import GetVideoMotion
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

__author__ = 'haoyi liang'


def main():
    demo = False
    is_sparse = 0
    sparse_corner_num = 1000
    moving_angle_thresh = [20, 10]
    stop_thresh = 1
    ite_num = 5
    resize_ratio = 0.5
    temporal_down_sample = 5
    roi = [[0, 600], [0, -1]]
    video_inf = GetVideoMotion(sparse_corner_num=sparse_corner_num, moving_angle_thresh=moving_angle_thresh,
                               stop_thresh=stop_thresh, ite_num=ite_num, is_sparse=is_sparse,
                               temporal_down_sample=temporal_down_sample, demo=demo)

    # video_path = '/mnt/scratch/sync_sd/car_record/road/20160524_01_road/20160513_zh_xf_card01/2016_0510_050549_001.mp4'
    # video_path = '/home/hoy/Haoyi/labeled_video/2016_0509_062938_036.mp4'
    video_path = '/mnt/scratch/sync_sd/car_record/both/20160526_01_both/20160526_wny_both/VDO_0002.mp4'
    frame_rate, v_point, vel, is_stop, cost, direction = video_inf.get_motion(video_path, roi, resize_ratio)

    plt.figure()
    plt.plot(signal.medfilt(direction))
    plt.title('direction')
    # np.savez('/home/hoy/Haoyi/labeled_video/2016_0509_062938_036.npz', frame_rate=frame_rate, v_point=v_point, vel=vel, is_stop=is_stop, cost=cost)



    plt.close('all')

if __name__ == '__main__':
    main()
