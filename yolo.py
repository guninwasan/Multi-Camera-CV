import cv2
import torch
import os


class YoloDetector:
    def __init__(self, model_path="yolov5s.pt", target_class_id=0):
        # Set the path to the YOLOv5 local directory
        yolo_dir = os.path.join(os.getcwd(), "yolov5")

        # Load the YOLOv5 model from local directory
        try:
            self.model = torch.hub.load(
                yolo_dir, "custom", path=model_path, source="local"
            )
            self.target_class_id = (
                target_class_id  # Class ID to detect, default is 'person'
            )
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            self.model = None

    def detect_objects(self, frame):
        if self.model is None:
            print("YOLOv5 model not loaded. Detection skipped.")
            return frame, []  # Return the frame as-is if model isn't loaded

        # Run inference on the frame
        results = self.model(frame)

        # Filter results to keep only the specified class
        detected_objects = results.xyxy[0][
            results.xyxy[0][:, -1] == self.target_class_id
        ]
        results.xyxy[0] = detected_objects

        # Extract bounding box details (x1, y1, x2, y2, confidence, class)
        boxes = [
            {
                "coordinates": (int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])),
                "confidence": float(obj[4]),
                "class_id": int(obj[5]),
            }
            for obj in detected_objects
        ]

        # Render bounding boxes and labels on the frame (optional)
        results.render()

        return frame, boxes  # Return both the modified frame and detected boxes
