import cv2
import time
import os
from aruco import ArucoDetector
from moving_object import OpticalFlow
from yolo import YoloDetector


class CombinedCVModel:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Failed to open camera.")

        self.aruco_detector = ArucoDetector()
        self.optical_flow = OpticalFlow()
        self.yolo_detector = YoloDetector()

        self.positions = []
        self.start_time = None
        self.human_interventions = 0

    def detect_and_track(self):
        self.start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            corners, ids = self.aruco_detector.detect_markers(frame)
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    self.positions.append((x_center, y_center))
                    cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"ID: {ids[i][0]}",
                        (x_center + 10, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    if edges[y_center, x_center] > 0:
                        cv2.putText(
                            frame,
                            "Collision Detected",
                            (x_center, y_center - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

            flow_bgr = self.optical_flow.calculate_flow(frame)
            cv2.imshow("Optical Flow", flow_bgr)

            results = self.yolo_detector.detect_objects(frame)
            for result in results.xyxy[0]:
                if result[-1] == 0:  # Assuming class 0 is 'person'
                    self.human_interventions += 1
                    break

            cv2.imshow("Combined CV Model", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        model = CombinedCVModel()
        model.detect_and_track()
    except Exception as e:
        print(f"An error occurred: {e}")
