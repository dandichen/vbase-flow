import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def set_accu(value):
    result = [["%.3f" % r] for r in value]
    return result


def load_kitti_result(result_path, method_name, gt_name, invalid_pairs):
    method_path = os.path.join(result_path, method_name)
    gt_path = os.path.join(result_path, gt_name)
    data = np.load(method_path+'.npz')
    gt = np.load(gt_path+'.npz')
    gt_v_point = gt['row_gt_v_point']
    gt_vel = gt['row_gt_vel']
    dect_v_point = data['v_point']
    dect_stop = data['is_stop']
    dect_vel = data['vel']
    x_origin = 610.0
    for _, ind in enumerate(invalid_pairs):
        gt_v_point[ind] = x_origin
        gt_vel[ind] = 0
        dect_v_point[ind] = x_origin
        dect_stop[ind] = 0
        dect_vel[ind] = 0

    gt_dir = np.arctan((gt_v_point[:, 0] - x_origin) / x_origin) / 3.14 * 180
    dect_v_point[np.where(dect_stop)] = x_origin
    dect_dir = np.arctan((dect_v_point[:, 0] - x_origin) / x_origin) / 3.14 * 180
    return gt_dir, gt_vel, dect_dir, dect_vel


def evaluate_kitti(gt_dir, gt_vel, dect_dir, evaluate_item, dir_bin, vel_bin_num, result_path, method_name, is_save):
    plt.figure()
    plt.plot(gt_dir, 'b')
    plt.plot(dect_dir, 'r')
    plt.title(evaluate_item[0])
    plt.waitforbuttonpress()

    evaluate_item_data = []
    dtype = [('sorting_item', float), ('for_evaluating', float)]
    values = zip(gt_dir, dect_dir)
    ana_data = np.array(values, dtype=dtype)
    evaluate_item_data.append(zip(*np.sort(ana_data, order='sorting_item')))
    values = zip(np.abs(gt_dir), np.abs(gt_dir - dect_dir))
    ana_data = np.array(values, dtype=dtype)
    evaluate_item_data.append(zip(*np.sort(ana_data, order='sorting_item')))
    values = zip(gt_vel, np.abs(gt_dir - dect_dir))
    ana_data = np.array(values, dtype=dtype)
    evaluate_item_data.append(zip(*np.sort(ana_data, order='sorting_item')))

    for i in range(1, 4):
        plt.figure()
        plt.title(evaluate_item[i])
        plt.plot(evaluate_item_data[i-1][0], 'b')
        plt.plot(evaluate_item_data[i-1][1], 'r')
        plt.waitforbuttonpress()

    cor_dir = stats.pearsonr(gt_dir, dect_dir)

    eva_by_dir_gt = []
    for i, _ in enumerate(dir_bin[:-1]):
        gt_dir = np.array(evaluate_item_data[1][0])
        dir_diff = np.array(evaluate_item_data[1][1])
        cur_bin = dir_diff[np.where(np.logical_and(gt_dir > dir_bin[i], gt_dir < dir_bin[i+1]))]
        eva_by_dir_gt.append(np.mean(cur_bin))

    vel_bin = np.linspace(0, np.max(gt_vel), vel_bin_num)
    eva_by_vel_gt = []
    for i, _ in enumerate(vel_bin[:-1]):
        gt_vel = np.array(evaluate_item_data[2][0])
        dir_diff = np.array(evaluate_item_data[2][1])
        cur_bin = dir_diff[np.where(np.logical_and(gt_vel > vel_bin[i], gt_vel < vel_bin[i + 1]))]
        eva_by_vel_gt.append(np.mean(cur_bin))

    if is_save:
        report_path = os.path.join(result_path, method_name)
        report_file = open(report_path+'.txt', 'w')
        report_file.write('direction correlation is {}\n'.format(cor_dir))
        report_file.write('bin edges by gt direction is' + repr(set_accu(dir_bin)) + '\n')
        report_file.write('ave direction diff is ' + repr(set_accu(eva_by_dir_gt)) + '\n')
        report_file.write('bin edges by gt velocity is' + repr(set_accu(vel_bin)) + '\n')
        report_file.write('ave direction diff is ' + repr(set_accu(eva_by_vel_gt)) + '\n')

        report_file.close()

    return cor_dir, eva_by_dir_gt, eva_by_vel_gt


def main():
    result_path = '/home/hoy/Haoyi/v_point_evaluation/'
    method_name = 'opencv_dense'
    gt_name = 'KITTI_gt'
    evaluate_item = ['ground truth and detected dir', 'dect dir sorted by gt dir',
                     'dir dif sorted by abs gt dir', 'dir dif sorted by abs gt vel']
    dir_bin = [0, 5, 10, 20, 90]
    vel_bin_num = 5
    is_save = False
    # 104: in the tunnel, totally dard
    # 169: almost static, but significant turning
    invalid_pairs = [104, 169]
    gt_dir, gt_vel, dect_dir, dect_vel\
        = load_kitti_result(result_path, method_name, gt_name, invalid_pairs)
    evaluate_kitti(gt_dir, gt_vel, dect_dir, evaluate_item, dir_bin, vel_bin_num, result_path, method_name, is_save)


if __name__ == '__main__':
    main()

