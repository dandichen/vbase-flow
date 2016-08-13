import cv2

from keypointPairs import KeypointPair

__author__ = 'Dandi Chen'

class ORB_point(KeypointPair):
    def __init__(self, bbox1, bbox2, kp1=None, kp2=None, des1=None, des2=None):
        KeypointPair.__init__(self, bbox1, bbox2, kp1, kp2, des1, des2)

    def get_keypoint(self, img1, img2):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        orb.setMaxFeatures(1200)
        orb.setScaleFactor(1.25)
        orb.setNLevels(6)
        orb.setEdgeThreshold(10)
        orb.setPatchSize(20)
        orb.setFastThreshold(8)

        # find the keypoints with ORB
        kp1 = orb.detect(img1, None)
        kp2 = orb.detect(img2, None)

        # compute the descriptors with ORB
        self.kp1, self.des1 = orb.compute(img1, kp1)
        self.kp2, self.des2 = orb.compute(img2, kp2)




