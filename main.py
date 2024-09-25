import cv2
import time
import numpy as np
import logging
import os
from datetime import datetime
from aruco import ArucoDetector
from moving_object import OpticalFlow
from yolo import YoloDetector


class CombinedCVModel:
    def __init__(self, camera_index=0):
        # Create directories for logs and reports
        self.base_dir = "logs_and_reports"
        os.makedirs(self.base_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = os.path.join(self.base_dir, self.timestamp)
        os.makedirs(self.report_dir, exist_ok=True)

        # Configure logging to file within the report directory
        log_file_path = os.path.join(self.report_dir, "cv_model.log")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logging.error("Failed to open camera.")
            raise Exception("Failed to open camera.")

        self.aruco_detector = ArucoDetector()
        self.optical_flow = OpticalFlow()
        self.yolo_detector = YoloDetector()

        # Initialize variables for tracking
        self.positions = []
        self.start_time = None
        self.human_interventions = 0
        self.errors = []

    def detect_and_track(self):
        self.start_time = time.time()
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                error_msg = "Failed to grab frame."
                logging.error(error_msg)
                self.errors.append(error_msg)
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
                    logging.info(commentary_text)
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
                        logging.warning(collision_text)
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
                    logging.info("Human intervention detected.")
                    break

            # Show the result frame
            cv2.imshow("Combined CV Model", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.generate_report()
        self.generate_error_report()

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

        report_path = os.path.join(self.report_dir, "score_report.txt")
        with open(report_path, "w") as f:
            f.write("--- Combined CV Model Report ---\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
        logging.info(f"Score report generated: {report_path}")

    def generate_error_report(self):
        if self.errors:
            error_report_path = os.path.join(self.report_dir, "error_report.txt")
            with open(error_report_path, "w") as f:
                f.write("Error Report\n")
                f.write("============\n")
                for error in self.errors:
                    f.write(f"{error}\n")
            logging.info(f"Error report generated: {error_report_path}")


if __name__ == "__main__":
    try:
        # Create an instance of the combined CV model
        model = CombinedCVModel()
        # Start detection and tracking
        model.detect_and_track()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        error_report_path = os.path.join(
            "logs_and_reports",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            "error_report.txt",
        )
        os.makedirs(os.path.dirname(error_report_path), exist_ok=True)
        with open(error_report_path, "w") as f:
            f.write("Error Report\n")
            f.write("============\n")
            f.write(f"An error occurred: {e}\n")
