import cv2
import numpy as np

__author__ = 'Dandi Chen'

def read_img(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2

def get_img_size(img):
    height, width = img.shape[:2]
    return width, height

def color2gray(img1, img2):
    prvs = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return prvs, next


class Flow(object):
    def __init__(self, width=480, height=320):
        self.val = np.zeros((width, height))
        self.x_val = 0
        self.y_val = 0

    def flow_decompose(self):
        self.x_val = self.val[:, :, 0]
        self.y_val = self.val[:, :, 1]

