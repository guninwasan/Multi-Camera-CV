import cv2
import torch
import os


class YoloDetector:
    def __init__(self):
        # Set the path to the YOLOv5 local directory
        yolo_dir = os.path.join(os.getcwd(), "yolov5")

        # Load the YOLOv5 model from local directory
        self.model = torch.hub.load(
            yolo_dir, "custom", path="yolov5s.pt", source="local"
        )

    def detect_objects(self, frame):
        results = self.model(frame)  # Run inference on the frame
        results.render()  # Updates the frame with bounding boxes and labels
        return results
