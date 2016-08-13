import numpy as np
from hid_inference import HidVarInf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from scipy import stats


def read_img(gt_path, img_num):
    img0_name = os.path.join(gt_path, str(img_num).zfill(6) + '_10.png')
    img1_name = os.path.join(gt_path, str(img_num).zfill(6) + '_11.png')
    img0 = cv2.imread(img0_name)
    img1 = cv2.imread(img1_name)
    return img0, img1


def main():
    gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2'
    row_ana_gt_path = '/home/hoy/Haoyi/Kitti/Scene_flow/extracted_data'
    demo = True
    is_sparse = 0
    test_segment = np.arange(17, 200)
    org_point = np.array([610, 173])
    sparse_corner_num = 500
    resize_ratio = 1
    moving_angle_thresh = [30, 10]
    stop_thresh = 1
    ite_num = 5
    v_point = np.zeros((len(test_segment), 2))
    cost = np.zeros((len(test_segment), 2))
    vel = np.zeros((len(test_segment), 1))
    is_stop = np.zeros((len(test_segment), 1))
    row_gt_v_point = np.zeros((len(test_segment), 2))
    direction = np.zeros((len(test_segment), 1))
    row_gt_vel = np.zeros((len(test_segment), 1))
    infer = HidVarInf(sparse_corner_num=sparse_corner_num, resize_ratio=resize_ratio, moving_angle_thresh=moving_angle_thresh,
                      stop_thresh=stop_thresh, ite_num=ite_num, is_sparse=is_sparse, demo=demo)
    for ind, img_num in enumerate(test_segment):
        print('image number is {}'.format(img_num))
        row_gt_data = np.load(os.path.join(row_ana_gt_path, '{:06d}'.format(img_num) + '.npz'))
        row_gt_v_point[ind, :] = row_gt_data['v_point']
        row_gt_vel[ind] = row_gt_data['ego_vel']
        print('the ground truth vanishing point is {} and {}'.format(row_gt_v_point[ind, 0], row_gt_v_point[ind, 1]))
        print("the ground truth direction is {}".format((np.arctan((row_gt_v_point[ind, 0] - org_point[0]) / org_point[0])) / 3.14 * 180))

        img0, img1 = read_img(gt_path, img_num)
        v_point[ind, :], vel[ind], is_stop[ind], cost[ind, :], direction[ind, :] = infer.get_motion(img0, img1, org_point)
        print('the detected vanishing point is {} and {}'.format(v_point[ind, 0], v_point[ind, 1]))
        print("the detected direction is {}".format((np.arctan((v_point[ind, 0] - org_point[0]) / org_point[0])) / 3.14 * 180))

    gt_dir = np.arctan((row_gt_v_point[:, 0] - org_point[0]) / org_point[0]) / 3.14 * 180
    plt.plot(gt_dir)
    plt.title('groundtruth direction')
    plt.waitforbuttonpress()
    plt.figure()
    dect_dir = np.arctan((v_point[:, 0] - org_point[0]) / org_point[0]) / 3.14 * 180
    plt.plot(dect_dir)
    plt.title('detected direction')
    plt.waitforbuttonpress()
    dir_linear_cor = stats.pearsonr(dect_dir, gt_dir)
    print('correlation between ground truth and detection is {}'.format(dir_linear_cor))

    plt.figure()
    plt.plot(row_gt_vel)
    plt.title('groundtruth velocity')
    plt.waitforbuttonpress()
    plt.figure()
    plt.plot(vel)
    plt.title('detected velocity')
    plt.waitforbuttonpress()
    vel_linear_cor = stats.pearsonr(vel, row_gt_vel)
    print('correlation between ground truth and detection is {}'.format(vel_linear_cor))



    dtype = [('groundtruth', float), ('detected', float)]
    values = zip(np.abs(gt_dir), np.abs(dect_dir))
    ana_data = np.array(values, dtype=dtype)
    sort_gt = zip(*np.sort(ana_data, order='groundtruth'))
    # np.savez('/home/hoy/Haoyi/v_point_evaluation/opencv_dense', v_point=v_point, vel=vel, is_stop=is_stop, cost=cost)
    # np.savez('KITTI_gt', row_gt_v_point=row_gt_v_point, row_gt_vel=row_gt_vel)

if __name__ == '__main__':
    main()

