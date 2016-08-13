from get_videomotion import GetVideoMotion
import matplotlib.pyplot as plt
import os
import numpy as np

__author__ = 'haoyi liang'


def main():
    demo = False
    is_sparse = 0
    roi = [[0, 600], [0, -1]]
    resize_ratio = 0.5
    sparse_corner_num = 1000
    moving_angle_thresh = [20, 10]
    stop_thresh = 1
    ite_num = 5
    temporal_down_sample = 5
    video_inf = GetVideoMotion(sparse_corner_num=sparse_corner_num, moving_angle_thresh=moving_angle_thresh,
                               stop_thresh=stop_thresh, ite_num=ite_num, is_sparse=is_sparse,
                               temporal_down_sample=temporal_down_sample, demo=demo)

    video_list ='/mnt/scratch/haoyiliang/tozhangke_20160607/readme.txt'
    save_path = '/mnt/scratch/haoyiliang/tozhangke_20160607'

    with open(video_list) as f:
        video_paths = f.readlines()
        video_paths.sort()
        for ind, video_path in enumerate(video_paths):
            split_path = video_path.split('/')
            video_name = split_path[-1].rstrip()
            first_dir = split_path[-2]
            print ind
            print video_name
            frame_rate, v_point, vel, is_stop, cost, direction = video_inf.get_motion(video_path.rstrip(), roi, resize_ratio)
            save_file = os.path.join(save_path, first_dir, video_name)
            np.savez(save_file+'.npz', frame_rate=frame_rate, v_point=v_point, vel=vel, is_stop=is_stop,
                     cost=cost, direction=direction)


if __name__ == '__main__':
    main()