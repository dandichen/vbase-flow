import cv2
import numpy as np

from bbox.boundingbox import BoundingBox
from bbox import grid

__author__ = 'Dandi Chen'

class Matcher(object):
    def __init__(self, key_pt_pair):
        self.key_pt_pair = key_pt_pair
        self.matches = cv2.DMatch()
        self.match_len = 0
        self.matchesMask = None

    def vis_matches(self, img1, img2, flag=0, show_start=0, show_end=50):
        valid_idx = np.where(self.matchesMask == True)[0]

        if self.key_pt_pair.bbox1.top_left_x == 0 and self.key_pt_pair.bbox1.top_left_y == 0 \
                and self.key_pt_pair.bbox2.top_left_x == 0 and self.key_pt_pair.bbox2.top_left_y == 0:
            height, width, _ = img1.shape
            outImg = np.zeros((width * 2, height))
            outImg = cv2.drawMatches(img1, self.key_pt_pair.kp1, img2, self.key_pt_pair.kp2,
                                     self.matches[show_start:show_end], outImg, flags=2)
            cv2.imshow('ORB matches', outImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            rows1, cols1, _ = img1.shape
            rows2, cols2, _ = img2.shape

            if flag == 0:   # horizontal visualization
                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
                out_img[:, 0:cols1, :] = img1
                out_img[:, cols1:cols1 + cols2, :] = img2

            else:           # vertical visualization
                out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
                out_img[0:rows1, :, :] = img1
                out_img[rows1:rows1 + rows2, :, :] = img2

            for mat_idx in range(len(self.matches[show_start:show_end])):
                if mat_idx in valid_idx or show_end - show_start == 1:
                    img1_idx = self.matches[show_start + mat_idx].queryIdx
                    img2_idx = self.matches[show_start + mat_idx].trainIdx

                    (x1, y1) = self.key_pt_pair.kp1[img1_idx].pt
                    (x2, y2) = self.key_pt_pair.kp2[img2_idx].pt

                    cv2.circle(out_img, (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                         int(y1 + self.key_pt_pair.bbox1.top_left_y)), 3, (255, 0, 0), 1)

                    if flag == 0:   # horizontal visualization
                        cv2.circle(out_img, (int(x2 + self.key_pt_pair.bbox2.top_left_x) + cols1,
                                             int(y2 + self.key_pt_pair.bbox2.top_left_y)), 3, (255, 0, 0), 1)
                    else:           # vertical visualization
                        cv2.circle(out_img, (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                             int(y2 + self.key_pt_pair.bbox2.top_left_y) + rows1), 3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))

                    if flag == 0:  # horizontal visualization
                        cv2.line(out_img,
                                 (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                  int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                 (int(x2 + self.key_pt_pair.bbox2.top_left_x) + cols1,
                                  int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                                 color[np.mod(mat_idx, 100)].tolist(), 1)
                    else:          # vertical visualization
                        cv2.line(out_img,
                                 (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                  int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                 (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                 int(y2 + self.key_pt_pair.bbox2.top_left_y) + rows1),
                                 color[np.mod(mat_idx, 100)].tolist(), 1)
                else:
                    continue

            # draw bounding box
            cv2.rectangle(out_img,
                          (int(self.key_pt_pair.bbox1.top_left_x),
                           int(self.key_pt_pair.bbox1.top_left_y)),
                          (int(self.key_pt_pair.bbox1.bottom_right_x),
                           int(self.key_pt_pair.bbox1.bottom_right_y)),
                          (0, 255, 0), 4)
            if flag == 0:  # horizontal visualization
                cv2.rectangle(out_img,
                              (int(self.key_pt_pair.bbox2.top_left_x) + cols1,
                               int(self.key_pt_pair.bbox2.top_left_y)),
                              (int(self.key_pt_pair.bbox2.bottom_right_x),
                               int(self.key_pt_pair.bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)
            else:          # vertical visualization
                cv2.rectangle(out_img,
                              (int(self.key_pt_pair.bbox2.top_left_x),
                               int(self.key_pt_pair.bbox2.top_left_y) + rows1),
                              (int(self.key_pt_pair.bbox2.bottom_right_x),
                               int(self.key_pt_pair.bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)

            # draw bounding box grid
            x_blk_size = 32
            y_blk_size = 32
            _, x_trans1, y_trans1, _, _, x_num1, y_num1 = \
                grid.grid_img(img1[int(self.key_pt_pair.bbox1.top_left_y):int(self.key_pt_pair.bbox1.bottom_right_y),
                              int(self.key_pt_pair.bbox1.top_left_x):int(self.key_pt_pair.bbox1.bottom_right_x)],
                              int(self.key_pt_pair.bbox1.width), int(self.key_pt_pair.bbox1.height),
                              x_blk_size, y_blk_size)
            box = BoundingBox()
            box.vis_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
                             int(self.key_pt_pair.bbox1.width), int(self.key_pt_pair.bbox1.height),
                             0, 0, int(self.key_pt_pair.bbox1.top_left_x), int(self.key_pt_pair.bbox1.top_left_y))

            _, x_trans2, y_trans2, _, _, x_num2, y_num2 = \
                grid.grid_img(img2[int(self.key_pt_pair.bbox2.top_left_y):int(self.key_pt_pair.bbox2.bottom_right_y),
                              int(self.key_pt_pair.bbox2.top_left_x):int(self.key_pt_pair.bbox2.bottom_right_x)],
                              int(self.key_pt_pair.bbox2.width), int(self.key_pt_pair.bbox2.height),
                              x_blk_size, y_blk_size)
            if flag == 0:  # horizontal visualization
                box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
                                 int(self.key_pt_pair.bbox2.width), int(self.key_pt_pair.bbox2.height),
                                 cols1, 0,
                                 int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
            else:           # vertical visualization
                box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
                                 int(self.key_pt_pair.bbox2.width), int(self.key_pt_pair.bbox2.height),
                                 0, rows1,
                                 int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
            cv2.imshow('ORB matches', out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def write_matches_overlap(self, img1, match_path, flag=0, show_start=0, show_end=50):
        valid_idx = np.where(self.matchesMask == True)[0]

        # 000000_10.png pair in original DMatch feature distance(weight = 0.5)
        # success_match = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        #                           24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 41, 42, 45, 46, 47, 48, 49, 51, 57, 58, 65,
        #                           66, 67, 68, 69, 81, 82, 92, 93, 94, 102, 104, 105, 110, 121, 122, 123, 124, 125, 126,
        #                           138, 153, 154, 155, 156, 157, 159, 179, 180, 190, 191, 192, 193, 195, 196, 198, 199,
        #                           212, 213, 225, 228, 229, 240, 241, 242, 258, 259, 260, 261, 270, 271, 272, 289, 291,
        #                           292, 294, 296, 307, 308, 309, 323, 324, 330, 331, 332, 358, 359, 367, 380, 381, 405, 406])

        out_img = img1.copy()
        for mat_idx in range(len(self.matches[show_start:show_end])):
            if mat_idx in valid_idx or show_end - show_start == 1:
                # if mat_idx in success_match:
                img1_idx = self.matches[show_start + mat_idx].queryIdx
                img2_idx = self.matches[show_start + mat_idx].trainIdx

                (x1, y1) = self.key_pt_pair.kp1[img1_idx].pt
                (x2, y2) = self.key_pt_pair.kp2[img2_idx].pt

                cv2.circle(out_img,
                           (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                            int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                           3, (255, 0, 0), 1)
                if flag == 0:  # horizontal visualization
                    cv2.circle(out_img,
                               (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                               3, (255, 0, 0), 1)
                else:  # vertical visualization
                    cv2.circle(out_img,
                               (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                               3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))
                if flag == 0:  # horizontal visualization
                    cv2.line(out_img,
                             (int(x1 + self.key_pt_pair.bbox1.top_left_x), int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                             (int(x2 + self.key_pt_pair.bbox2.top_left_x), int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                             color[np.mod(mat_idx, 100)].tolist(), 1)
                else:  # vertical visualization
                    cv2.line(out_img,
                             (int(x1 + self.key_pt_pair.bbox1.top_left_x), int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                             (int(x2 + self.key_pt_pair.bbox2.top_left_x), int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                             color[np.mod(mat_idx, 100)].tolist(), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                if np.mod(mat_idx, 2) == 0:
                    cv2.putText(out_img, str(mat_idx),
                                (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                 int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
                else:
                    cv2.putText(out_img, str(mat_idx),
                                (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                 int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                                font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
            else:
                continue

        # draw bounding box
        cv2.rectangle(out_img,
                      (int(self.key_pt_pair.bbox1.top_left_x), int(self.key_pt_pair.bbox1.top_left_y)),
                      (int(self.key_pt_pair.bbox1.bottom_right_x), int(self.key_pt_pair.bbox1.bottom_right_y)),
                      (0, 255, 0), 4)
        if flag == 0:  # horizontal visualization
            cv2.rectangle(out_img,
                          (int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y)),
                          (int(self.key_pt_pair.bbox2.bottom_right_x), int(self.key_pt_pair.bbox2.bottom_right_y)),
                          (0, 255, 0), 4)
        else:  # vertical visualization
            cv2.rectangle(out_img,
                          (int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y)),
                          (int(self.key_pt_pair.bbox2.bottom_right_x), int(self.key_pt_pair.bbox2.bottom_right_y)),
                          (0, 255, 0), 4)

            # # draw bounding box grid
            # x_blk_size = 32
            # y_blk_size = 32
            # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = \
            #     grid.grid_img(img1[int(self.key_pt_pair.bbox1.top_left_y):int(self.key_pt_pair.bbox1.bottom_right_y),
            #                   int(self.key_pt_pair.bbox1.top_left_x):int(self.key_pt_pair.bbox1.bottom_right_x)],
            #                   int(self.key_pt_pair.bbox1.bottom_right_x - self.key_pt_pair.bbox1.top_left_x),
            #                   int(self.key_pt_pair.bbox1.bottom_right_y - self.key_pt_pair.bbox1.top_left_y),
            #                   x_blk_size, y_blk_size)
            #
            # box = BoundingBox()
            # box.write_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
            #                    int(self.key_pt_pair.bbox1.bottom_right_x - self.key_pt_pair.bbox1.top_left_x),
            #                    int(self.key_pt_pair.bbox1.bottom_right_y - self.key_pt_pair.bbox1.top_left_y), 0, 0,
            #                    int(self.key_pt_pair.bbox1.top_left_x), int(self.key_pt_pair.bbox1.top_left_y))
            #
            # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = \
            #     grid.grid_img(img2[int(self.key_pt_pair.bbox2.top_left_y):int(self.key_pt_pair.bbox2.bottom_right_y),
            #                   int(self.key_pt_pair.bbox2.top_left_x):int(self.key_pt_pair.bbox2.bottom_right_x)],
            #                   int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                   int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y),
            #                   x_blk_size, y_blk_size)
            # if flag == 0:  # horizontal visualization
            #     box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                        int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                        int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y), cols1, 0,
            #                        int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
            # else:  # vertical visualization
            #     box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                        int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                        int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y), 0, rows1,
            #                        int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
        cv2.imwrite(match_path, out_img)

    def write_matches(self, img1, img2, match_path, flag=0, show_start=0, show_end=50):
        valid_idx = np.where(self.matchesMask == True)[0]

        # 000000_10.png pair in original DMatch feature distance(weight = 0.5)
        # success_match = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        #                           24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 41, 42, 45, 46, 47, 48, 49, 51, 57, 58, 65,
        #                           66, 67, 68, 69, 81, 82, 92, 93, 94, 102, 104, 105, 110, 121, 122, 123, 124, 125, 126,
        #                           138, 153, 154, 155, 156, 157, 159, 179, 180, 190, 191, 192, 193, 195, 196, 198, 199,
        #                           212, 213, 225, 228, 229, 240, 241, 242, 258, 259, 260, 261, 270, 271, 272, 289, 291,
        #                           292, 294, 296, 307, 308, 309, 323, 324, 330, 331, 332, 358, 359, 367, 380, 381, 405, 406])
        if self.key_pt_pair.bbox1.top_left_x == 0 and self.key_pt_pair.bbox1.top_left_y == 0 \
                and self.key_pt_pair.bbox2.top_left_x == 0 and self.key_pt_pair.bbox2.top_left_y == 0:
            height, width, _ = img1.shape
            outImg = np.zeros((width * 2, height))
            outImg = cv2.drawMatches(img1, self.key_pt_pair.kp1, img2, self.key_pt_pair.kp2, self.matches, outImg, flags=2)
            cv2.imwrite(match_path, outImg)
        else:
            rows1, cols1, _ = img1.shape
            rows2, cols2, _ = img2.shape

            if flag == 0:  # horizontal visualization
                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
                out_img[:, 0:cols1, :] = img1
                out_img[:, cols1:cols1 + cols2, :] = img2
            else:  # vertical visualization
                out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
                out_img[0:rows1, :, :] = img1
                out_img[rows1:rows1 + rows2, :, :] = img2

            for mat_idx in range(len(self.matches[show_start:show_end])):
                if mat_idx in valid_idx or show_end - show_start == 1:
                    # if mat_idx in success_match:
                    img1_idx = self.matches[show_start + mat_idx].queryIdx
                    img2_idx = self.matches[show_start + mat_idx].trainIdx

                    (x1, y1) = self.key_pt_pair.kp1[img1_idx].pt
                    (x2, y2) = self.key_pt_pair.kp2[img2_idx].pt

                    cv2.circle(out_img,
                               (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                               3, (255, 0, 0), 1)
                    if flag == 0:  # horizontal visualization
                        cv2.circle(out_img,
                                   (int(x2 + self.key_pt_pair.bbox2.top_left_x) + cols1,
                                    int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                                   3, (255, 0, 0), 1)
                    else:  # vertical visualization
                        cv2.circle(out_img,
                                   (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                    int(y2 + self.key_pt_pair.bbox2.top_left_y) + rows1),
                                   3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    if flag == 0:  # horizontal visualization
                        cv2.line(out_img,
                                 (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                  int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                 (int(x2 + self.key_pt_pair.bbox2.top_left_x) + cols1,
                                  int(y2 + self.key_pt_pair.bbox2.top_left_y)),
                                 color[np.mod(mat_idx, 100)].tolist(), 1)
                    else:  # vertical visualization
                        cv2.line(out_img,
                                 (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                  int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                 (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                  int(y2 + self.key_pt_pair.bbox2.top_left_y) + rows1),
                                 color[np.mod(mat_idx, 100)].tolist(), 1)
                        font = cv2.FONT_HERSHEY_PLAIN
                        if np.mod(mat_idx, 2) == 0:
                            cv2.putText(out_img, str(mat_idx),
                                        (int(x1 + self.key_pt_pair.bbox1.top_left_x),
                                         int(y1 + self.key_pt_pair.bbox1.top_left_y)),
                                        font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
                        else:
                            cv2.putText(out_img, str(mat_idx),
                                        (int(x2 + self.key_pt_pair.bbox2.top_left_x),
                                         int(y2 + self.key_pt_pair.bbox2.top_left_y) + rows1),
                                        font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
                else:
                    continue

            # draw bounding box
            cv2.rectangle(out_img,
                          (int(self.key_pt_pair.bbox1.top_left_x), int(self.key_pt_pair.bbox1.top_left_y)),
                          (int(self.key_pt_pair.bbox1.bottom_right_x), int(self.key_pt_pair.bbox1.bottom_right_y)),
                          (0, 255, 0), 4)
            if flag == 0:  # horizontal visualization
                cv2.rectangle(out_img,
                              (int(self.key_pt_pair.bbox2.top_left_x) + cols1,
                               int(self.key_pt_pair.bbox2.top_left_y)),
                              (int(self.key_pt_pair.bbox2.bottom_right_x),
                               int(self.key_pt_pair.bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)
            else:  # vertical visualization
                cv2.rectangle(out_img,
                              (int(self.key_pt_pair.bbox2.top_left_x),
                               int(self.key_pt_pair.bbox2.top_left_y) + rows1),
                              (int(self.key_pt_pair.bbox2.bottom_right_x),
                               int(self.key_pt_pair.bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)

            # # draw bounding box grid
            # x_blk_size = 32
            # y_blk_size = 32
            # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = \
            #     grid.grid_img(img1[int(self.key_pt_pair.bbox1.top_left_y):int(self.key_pt_pair.bbox1.bottom_right_y),
            #                   int(self.key_pt_pair.bbox1.top_left_x):int(self.key_pt_pair.bbox1.bottom_right_x)],
            #                   int(self.key_pt_pair.bbox1.bottom_right_x - self.key_pt_pair.bbox1.top_left_x),
            #                   int(self.key_pt_pair.bbox1.bottom_right_y - self.key_pt_pair.bbox1.top_left_y),
            #                   x_blk_size, y_blk_size)
            # box = BoundingBox()
            # box.write_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
            #                    int(self.key_pt_pair.bbox1.bottom_right_x - self.key_pt_pair.bbox1.top_left_x),
            #                    int(self.key_pt_pair.bbox1.bottom_right_y - self.key_pt_pair.bbox1.top_left_y),
            #                    0, 0,
            #                    int(self.key_pt_pair.bbox1.top_left_x), int(self.key_pt_pair.bbox1.top_left_y))
            #
            # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = \
            #     grid.grid_img(img2[int(self.key_pt_pair.bbox2.top_left_y):int(self.key_pt_pair.bbox2.bottom_right_y),
            #                   int(self.key_pt_pair.bbox2.top_left_x):int(self.key_pt_pair.bbox2.bottom_right_x)],
            #                   int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                   int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y),
            #                   x_blk_size, y_blk_size)
            # if flag == 0:  # horizontal visualization
            #     box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                        int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                        int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y),
            #                        cols1, 0,
            #                        int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
            # else:  # vertical visualization
            #     box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                        int(self.key_pt_pair.bbox2.bottom_right_x - self.key_pt_pair.bbox2.top_left_x),
            #                        int(self.key_pt_pair.bbox2.bottom_right_y - self.key_pt_pair.bbox2.top_left_y),
            #                        0, rows1,
            #                        int(self.key_pt_pair.bbox2.top_left_x), int(self.key_pt_pair.bbox2.top_left_y))
            cv2.imwrite(match_path, out_img)
