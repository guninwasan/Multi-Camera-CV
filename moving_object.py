import cv2
import numpy as np


class OpticalFlow:
    def __init__(self):
        self.prev_gray = None
        self.hsv = None

    def initialize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.hsv = np.zeros_like(frame)
        self.hsv[..., 1] = 255

    def calculate_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.initialize(frame)
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = angle * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
        self.prev_gray = gray
        return flow_bgr
