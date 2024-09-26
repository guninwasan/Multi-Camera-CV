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
        cv2.namedWindow("Combined CV Model", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Combined CV Model", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

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
                        (0, 255, 255),  # Brighter color (yellow)
                        2,
                    )
                    # Add commentary text
                    commentary_text = (
                        f"ArUco Marker ID {ids[i][0]} at ({x_center}, {y_center})"
                    )
                    cv2.putText(
                        frame,
                        commentary_text,
                        (10, 30 + 30 * i),  # Position the text below each other
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),  # Brighter color (yellow)
                        2,
                    )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Mark detected edges (walls) on the frame
            frame[edges != 0] = [0, 0, 255]  # Mark edges in red

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

            # Check for collision with the wall using edge detection
            frame_height, frame_width = frame.shape[:2]
            collision_detected = False
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    if edges[y_center, x_center] > 0:
                        collision_detected = True
                        print(
                            f"Collision detected at position: ({x_center}, {y_center})"
                        )
                        cv2.putText(
                            frame,
                            "Wall Collision Detected",
                            (x_center, y_center - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )
                        break

            if not collision_detected and ids is None:
                # Use optical flow as a fail-proof measure
                flow_bgr = self.optical_flow.calculate_flow(frame)
                alpha = 0.5  # Transparency factor
                frame = cv2.addWeighted(flow_bgr, alpha, frame, 1 - alpha, 0)

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
        video_source = "/Users/guninwasan/Downloads/CV-System/Screen Recording 2024-09-25 at 18.59.51.mov"
        model = CombinedCVModel(video_source)
        model.detect_and_track()
    except Exception as e:
        print(f"An error occurred: {e}")
