import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.insert(1, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib')

sys.path.append('/mnt/scratch/third-party-packages/mxnet/python')

import os
import numpy as np
import matplotlib.pyplot as plt

import video_tools as vtool
import feature.flow_base as flow_base
import learning.MotionInference.hid_inference as hid_inf
import visualization.velocity_vector as vel_vis
import visualization.vanishing_point as vPt_vis

from feature.dense_flow import DenseFlow


__author__ = 'Dandi Chen'

def main():
    video_path = '/mnt/scratch/sync_sd/car_record/road/20160524_01_road/20160513_zh_xf_card02/2016_0508_021615_022.mp4'
    video2img_path = '/mnt/scratch/DandiChen/test/data'
    vis_path = '/mnt/scratch/DandiChen/test/visualization'
    split_step = 30
    vis_step = 30

    # extract frames from videos
    out_frames, out_frames_ID = vtool.vidoe2img(video_path, split_step)

    for img_idx in range(len(out_frames)):
        plt.imsave(os.path.join(video2img_path + '/' + str(img_idx).zfill(6) + '.png'), out_frames[img_idx])

    # compute flow
    img_num = len([name for name in os.listdir(video2img_path) if os.path.isfile(os.path.join(video2img_path, name))]) - 1
    flow_num = img_num - 1  # continuous two frames

    for img in range(flow_num - 1):
        print ''
        print 'img number: ', img

        img_path1 = os.path.join(video2img_path, str(img).zfill(6) + '.png')
        img_path2 = os.path.join(video2img_path, str(img+1).zfill(6) + '.png')
        img1, img2 = flow_base.read_img(img_path1, img_path2)

        width, height = flow_base.get_img_size(img1)
        prvs, next = flow_base.color2gray(img1, img2)

        den_flow = DenseFlow(width, height)
        den_flow.compute(prvs, next)

        # visualize flow
        vel_img = vel_vis.plot_velocity_vector(den_flow.val, vis_step)
        plt.imsave(os.path.join(vis_path + str('/velocity'), str(img).zfill(6) + '.png'), vel_img)

    # compute vanishing point
    pt = np.zeros((img_num, 2))
    cost = np.zeros((img_num, 2))
    vel = np.zeros((img_num, 1))
    is_stop = np.zeros((img_num, 1))
    direction = np.zeros((img_num, 1))

    infer = hid_inf.HidVarInf(sparse_corner_num=100, moving_angle_thresh=[30, 10], \
                              stop_thresh=1, ite_num=5, is_sparse=0, demo=False)

    for idx in range(flow_num - 1):
        print ''
        print 'img_num = ', idx

        img_path1 = os.path.join(video2img_path, str(idx).zfill(6) + '.png')
        img_path2 = os.path.join(video2img_path, str(idx+1).zfill(6) + '.png')
        img1, img2 = flow_base.read_img(img_path1, img_path2)
        pt[idx, :], vel[idx], is_stop[idx], cost[idx, :], direction[idx] = infer.get_motion(img1, img2)

        # visualize vanishing point
        fig = vPt_vis.plot_vanishing_point(pt[idx, 0], pt[idx, 1], img1)
        fig.savefig(os.path.join(vis_path + str('/vanishing_point'), str(idx).zfill(6) + '.png'))

if __name__ == '__main__':
    main()
