import cv2
import numpy as np

__author__ = 'Dandi Chen'


class Flow(object):
    def __init__(self, matcher, width=1242, height=375):
        self.width = width
        self.height = height
        self.matcher = matcher
        self.val_x = np.zeros((self.height, self.width))
        self.val_y = np.zeros((self.height, self.width))
        self.mask = np.zeros((self.height, self.width), dtype=bool)

    def normalize_flow(self):
        val_x_nr = (self.val_x - np.min(self.val_x)) / (np.float_(np.max(self.val_x) - np.min(self.val_x)))
        val_y_nr = (self.val_y - np.min(self.val_y)) / (np.float_(np.max(self.val_y) - np.min(self.val_y)))
        return val_x_nr, val_y_nr

    def reshape_vec(self, width, height):
        flow_vec_x = np.reshape(self.val_x, width * height)
        flow_vec_y = np.reshape(self.val_y, width * height)
        return flow_vec_x, flow_vec_y

    def compute(self, match_path):
        flow = Flow()

        for mat_idx in range(self.matcher.match_len):
            img1_idx = self.matcher.matches[mat_idx].queryIdx
            img2_idx = self.matcher.matches[mat_idx].trainIdx

            (x1, y1) = self.matcher.key_pt_pair.kp1[img1_idx].pt
            (x2, y2) = self.matcher.key_pt_pair.kp2[img2_idx].pt

            delta_x = (x2 + self.matcher.key_pt_pair.bbox2.top_left_x) - (x1 + self.matcher.key_pt_pair.bbox1.top_left_x)
            delta_y = (y2 + self.matcher.key_pt_pair.bbox2.top_left_y) - (y1 + self.matcher.key_pt_pair.bbox1.top_left_y)

            # flow has been defined in first frame
            flow.val_x[int(y1 + self.matcher.key_pt_pair.bbox1.top_left_y)][int(x1 + self.matcher.key_pt_pair.bbox1.top_left_x)] = delta_x
            flow.val_y[int(y1 + self.matcher.key_pt_pair.bbox1.top_left_y)][int(x1 + self.matcher.key_pt_pair.bbox1.top_left_x)] = delta_y
            flow.mask[int(y1 + self.matcher.key_pt_pair.bbox1.top_left_y)][int(x1 + self.matcher.key_pt_pair.bbox1.top_left_x)] = True

        # np.savez(match_path, val_x=self.val_x, val_y=self.val_y, mask=self.mask)

    # visualization
    def write_flow2match_mask(self, img1, img2, width, height, vel_path, bbox1, bbox2, step=3):
        rows1, cols1, _ = img1.shape
        rows2, cols2, _ = img2.shape

        out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        out_img[0:rows1, :, :] = img1
        out_img[rows1:rows1 + rows2, :, :] = img2

        # draw bounding box
        cv2.rectangle(out_img,
                      (int(bbox1.top_left_x), int(bbox1.top_left_y)),
                      (int(bbox1.bottom_right_x), int(bbox1.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.rectangle(out_img,
                      (int(bbox2.top_left_x), int(bbox2.top_left_y) + rows1),
                      (int(bbox2.bottom_right_x), int(bbox2.bottom_right_y) + rows1),
                      (0, 255, 0), 4)

        for j in range(0, width - step, step):
            for i in range(0, height - step, step):
                if self.mask[i, j] == True:
                    cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                    cv2.circle(out_img,
                               (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j])) + rows1),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    cv2.line(out_img, (j, i),
                             (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j])) + rows1),
                             color[np.mod(i + j, 100)].tolist(), 1)
        cv2.imwrite(vel_path, out_img)

    def write_flow2match_overlap_mask(self, img, width, height, vel_path, bbox1, bbox2, step=3):
        out_img = img.copy()

        # draw bounding box
        cv2.rectangle(out_img,
                      (int(bbox1.top_left_x), int(bbox1.top_left_y)),
                      (int(bbox1.bottom_right_x), int(bbox1.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.rectangle(out_img,
                      (int(bbox2.top_left_x), int(bbox2.top_left_y)),
                      (int(bbox2.bottom_right_x), int(bbox2.bottom_right_y)),
                      (0, 255, 0), 4)

        for j in range(0, width - step, step):
            for i in range(0, height - step, step):
                if self.mask[i, j] == True:
                    cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                    cv2.circle(out_img,
                               (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    cv2.line(out_img, (j, i), (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                             color[np.mod(i + j, 100)].tolist(), 1)
        cv2.imwrite(vel_path, out_img)

    def write_velocity_vector_compare(self, img, flow_gt, vel_path, step1=10, step2=10):
        # white background
        vel_img = np.ones((self.height, self.width, 3), dtype=np.float64)*255
        for j in range(0, self.width - step1, step1):
            for i in range(0, self.height - step1, step1):
                # cv2.arrowedLine(vel_img, (j, i),
                #                 (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                #                 (255, 0, 0), 2)
                cv2.arrowedLine(vel_img, (j, i),
                                (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
                                (0, 0, 150), 2)

        #         cv2.arrowedLine(img, (j, i),
        #                         (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
        #                         (255, 0, 0), 2)
        #         cv2.arrowedLine(img, (j, i),
        #                         (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
        #                         (0, 0, 150), 2)
        # for j in range(0, self.width - step2, step2):
            for i in range(0, self.height - step2, step2):
                cv2.arrowedLine(vel_img, (j, i),
                                (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                                (255, 0, 0), 2)
                # cv2.arrowedLine(vel_img, (j, i),
                #                 (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
                #                 (0, 0, 150), 2)
        cv2.imwrite(vel_path, vel_img)

    def write_velocity_vector_compare_mask(self, flow_gt, vel_path, step1=10, step2=10):
        # white background
        vel_img = np.ones((self.height, self.width, 3), dtype=np.float64)*255
        for j in range(0, self.width - step1, step1):
            for i in range(0, self.height - step1, step1):
                    cv2.arrowedLine(vel_img, (j, i),
                                    (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
                                    (0, 0, 150), 2)

        for j in range(0, self.width - step2, step2):
            for i in range(0, self.height - step2, step2):
                if self.mask[i, j]:
                    cv2.arrowedLine(vel_img, (j, i),
                                    (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                                    (255, 0, 0), 2)
        cv2.imwrite(vel_path, vel_img)