import os
import timeit
import numpy as np

import keypoint_detection.keypointPairs as keypoint_pairs

from bbox.boundingbox import BoundingBox
from keypoint_detection.ORB import ORB_point
from keypoint_matching.brute_force import BruteForceMatcher

__author__ = 'Dandi Chen'


img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

eval_path = '/mnt/scratch/DandiChen/keypoint/KITTI/pipeline/confidence/'
kp_path = eval_path + 'keypoint/'
match_path = eval_path + 'matches/000000_10_customized/box/weighted_dis/w-0.5/'
match_path_overlap = match_path + 'overlap/'
match_path_non_overlap = match_path + 'non_overlap/'
flow_path = eval_path + 'flow/'
velocity_path = eval_path + 'velocity/'


img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

pair_num = img_num/2
# pair_num = 2
# pair_num = 1

t = []

for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    # start = timeit.default_timer()
    img1, img2 = keypoint_pairs.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    start = timeit.default_timer()

    # bounding box coordinates
    bbox1 = BoundingBox(154.07749939, 181.342102051, 405.574401855, 305.924407959)
    bbox2 = BoundingBox(0.0, 156.604873657, 353.453063965, 351.0)

    # ORB keypoint
    orb = ORB_point(bbox1, bbox2)
    orb.get_keypoint(img1[int(bbox1.top_left_y):int(bbox1.bottom_right_y),
                     int(bbox1.top_left_x):int(bbox1.bottom_right_x)],
                     img2[int(bbox2.top_left_y):int(bbox2.bottom_right_y),
                     int(bbox2.top_left_x):int(bbox2.bottom_right_x)])
    t_kp = timeit.default_timer()
    print 'keypoint extraction time:', t_kp - start
    print 'keypoint number:', len(orb.kp1), len(orb.kp2)
    print ''

    # orb.vis_pt_pairs(img1, img2)
    # orb.write_pt_pairs(img1, img2, os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
    #                    os.path.join(kp_path, str(img).zfill(6) + '_11.png'))

    t_matcher_0 = timeit.default_timer()
    # BFMatcher
    bfm = BruteForceMatcher(orb)
    bfm.get_matcher()

    t_matcher = timeit.default_timer()
    print 'matcher number:', bfm.match_len
    print 'matcher time :', t_matcher - t_matcher_0
    print ''

    t_good_matcher_0 = timeit.default_timer()
    # find homography
    bfm.get_good_matcher()           # Lowe's good feature threshold criteria(feature similarity distance)
    t_good_matcher = timeit.default_timer()
    print 'good matcher number:', bfm.match_len
    print 'good matcher time:', t_good_matcher - t_good_matcher_0
    print ''

    t_wgt_matcher_0 = timeit.default_timer()
    # confidence evaluation
    bfm.get_wgt_dis_matcher()
    bfm.get_homography()
    # bfm.vis_matches(img1, img2, 1)
    # bfm.write_matches(img1, img2, os.path.join(match_path, str(img).zfill(6) + '_10_non_overlap_match.png'), 1)
    # bfm.write_matches_overlap(img1, os.path.join(match_path, str(img).zfill(6) + '_10_overlap_match.png'), 1)

    t_wgt_matcher = timeit.default_timer()
    print 'good weighted matcher number:', bfm.match_len
    print 'good weighted matcher time:', t_wgt_matcher - t_wgt_matcher_0
    print ''

    end = timeit.default_timer()
    print 'total time = ', end - start
    t.append([end - start])

print ''
print 'ave time = ', np.mean(t)
