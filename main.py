import cv2
import time
from aruco import ArucoDetector
from moving_object import OpticalFlow
from yolo import YoloDetector
import numpy as np


class CombinedCVModel:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)  # Open video capture
        self.aruco_detector = ArucoDetector()
        self.optical_flow = OpticalFlow()
        self.yolo_detector = YoloDetector()

        # Initialize variables for tracking
        self.positions = []
        self.start_time = None
        self.human_interventions = 0

    def detect_and_track(self):
        self.start_time = time.time()
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # ArUco marker detection
            corners, ids = self.aruco_detector.detect_markers(frame)
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    self.positions.append((x_center, y_center))
                    commentary_text = f"ArUco Marker ID {ids[i][0]} detected at position: ({x_center}, {y_center})"
                    print(commentary_text)
                    cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

            # Edge detection to find walls
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Check for collision
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    if edges[y_center, x_center] > 0:
                        collision_text = f"Collision detected for ArUco Marker ID {ids[i][0]} at position: ({x_center}, {y_center})"
                        print(collision_text)
                        cv2.putText(
                            frame,
                            "Collision Detected",
                            (x_center, y_center - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

            # Optical flow to monitor movement
            flow_bgr = self.optical_flow.calculate_flow(frame)
            cv2.imshow("Optical Flow", flow_bgr)

            # YOLOv5 to detect human intervention
            results = self.yolo_detector.detect_objects(frame)
            for result in results.xyxy[0]:
                if result[-1] == 0:  # Assuming class 0 is 'person'
                    self.human_interventions += 1
                    break

            # Show the result frame
            cv2.imshow("Combined CV Model", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.generate_report()

    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def generate_report(self):
        total_time = time.time() - self.start_time
        total_distance = sum(
            self.calculate_distance(self.positions[i], self.positions[i + 1])
            for i in range(len(self.positions) - 1)
        )
        average_speed = total_distance / total_time if total_time > 0 else 0

        report = {
            "Total Time Elapsed": total_time,
            "Total Distance Traveled": total_distance,
            "Average Speed": average_speed,
            "Number of Human Interventions": self.human_interventions,
        }

        print("\n--- Combined CV Model Report ---")
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Create an instance of the combined CV model
    model = CombinedCVModel()
    # Start detection and tracking
    model.detect_and_track()
