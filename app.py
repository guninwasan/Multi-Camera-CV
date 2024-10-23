import argparse
from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import threading
import numpy as np
from aruco import ArucoDetector
from moving_object import OpticalFlow

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables
cap = cv2.VideoCapture(
    "/Users/guninwasan/Downloads/CV-System/vid.mp4"
)  # Update video source
start_time = None
timer_running = False
robot_positions = []  # Store robot positions to calculate average speed
team_name = "Team"

# Lock for thread-safe operations
lock = threading.Lock()

# Initialize detectors
aruco_detector = ArucoDetector()
optical_flow = OpticalFlow()


# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to detect the robot and calculate position and speed
def detect_robot(frame):
    global robot_positions

    # Attempt to detect ArUco marker
    corners, ids = aruco_detector.detect_markers(frame)

    if ids is not None:
        # If ArUco markers are found, use the center of the marker as the robot's position
        c = corners[0][0]  # Assume we're tracking one robot
        x_center = int(c[:, 0].mean())
        y_center = int(c[:, 1].mean())
        current_position = (x_center, y_center)

        # Display robot position on frame
        cv2.circle(frame, current_position, 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Position: ({x_center}, {y_center})",
            (x_center + 10, y_center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
    else:
        # If no ArUco marker is detected, fall back to optical flow
        flow_bgr = optical_flow.calculate_flow(frame)
        alpha = 0.5  # Transparency factor
        frame = cv2.addWeighted(flow_bgr, alpha, frame, 1 - alpha, 0)

        if robot_positions:
            current_position = robot_positions[-1]  # Use last known position
        else:
            current_position = (0, 0)  # Default position if no prior data

    # Append current position to the list for speed calculation
    robot_positions.append((current_position, time.time()))

    return frame, current_position


# Function to calculate average speed
def calculate_average_speed():
    if len(robot_positions) < 2:
        return 0.0

    total_distance = 0.0
    total_time = 0.0

    for i in range(1, len(robot_positions)):
        p1, t1 = robot_positions[i - 1]
        p2, t2 = robot_positions[i]
        distance = calculate_distance(p1, p2)
        time_diff = t2 - t1

        total_distance += distance
        total_time += time_diff

    # Avoid division by zero
    if total_time == 0:
        return 0.0

    average_speed = total_distance / total_time
    return average_speed


# Function to generate frames
def generate_frames():
    global start_time, timer_running, robot_positions

    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            # Detect robot and get position
            frame, current_position = detect_robot(frame)

            # Timer display
            if timer_running and start_time is not None:
                elapsed_time = time.time() - start_time
                cv2.rectangle(frame, (10, 10, 300, 60), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Elapsed Time: {elapsed_time:.2f} s",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Display average speed
            average_speed = calculate_average_speed()
            cv2.putText(
                frame,
                f"Avg Speed: {average_speed:.2f} px/s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/start_stop_timer", methods=["POST"])
def start_stop_timer():
    global start_time, timer_running
    with lock:
        if timer_running:
            timer_running = False
        else:
            timer_running = True
            if start_time is None:
                start_time = time.time()
    return jsonify(status="success")


@app.route("/reset", methods=["POST"])
def reset():
    global start_time, timer_running, robot_positions
    with lock:
        start_time = None
        timer_running = False
        robot_positions = []  # Reset positions
    return jsonify(status="success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the application.")
    args = parser.parse_args()

    try:
        app.run(debug=True, threaded=True)  # Run in threaded mode for performance
    except Exception as e:
        print(f"Failed to start Flask server: {e}")
