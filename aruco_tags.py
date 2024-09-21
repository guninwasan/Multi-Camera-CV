import cv2
import numpy as np


class RoverDetector:
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_6X6_250, camera_index=0):
        # Load the specified ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.cap = cv2.VideoCapture(camera_index)  # Open video capture

    def detect_rover(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert the image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers in the image
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            # If markers are detected
            if ids is not None:
                # Draw the detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, corner in enumerate(corners):
                    # Get the center of the ArUco marker
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    print(
                        f"ArUco Marker ID {ids[i][0]} detected at position: ({x_center}, {y_center})"
                    )
                    # Optionally, draw a circle in the center of the ArUco marker
                    cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

            # Edge detection to find walls
            edges = cv2.Canny(gray, 50, 150)

            # Check for collision
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
                    if edges[y_center, x_center] > 0:
                        print(
                            f"Collision detected for ArUco Marker ID {ids[i][0]} at position: ({x_center}, {y_center})"
                        )
                        cv2.putText(
                            frame,
                            "Collision Detected",
                            (x_center, y_center - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                        )

            # Show the result frame
            cv2.imshow("Rover Detection", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything is done, release the capture and close windows
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create an instance of the rover detector
    detector = RoverDetector()
    # Start detection
    detector.detect_rover()
