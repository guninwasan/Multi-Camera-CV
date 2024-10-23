import cv2
from flask import Flask, render_template, request, jsonify, Response
import time
import threading
import numpy as np
from aruco import ArucoDetector
from moving_object import OpticalFlow

app = Flask(__name__)

# Global variables
cap = cv2.VideoCapture(0)
team_name = None
timer_running = False
start_time = None
lock = threading.Lock()

aruco_detector = ArucoDetector()
optical_flow = OpticalFlow()
robot_positions = []  # Store robot positions


# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to detect the robot
def detect_robot(frame):
    global robot_positions

    corners, ids = aruco_detector.detect_markers(frame)
    if ids is not None:
        c = corners[0][0]
        x_center = int(c[:, 0].mean())
        y_center = int(c[:, 1].mean())
        current_position = (x_center, y_center)

        # Draw circle and display position
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
        # Optical flow when no marker is detected
        flow_bgr = optical_flow.calculate_flow(frame)
        alpha = 0.5  # Transparency factor
        frame = cv2.addWeighted(flow_bgr, alpha, frame, 1 - alpha, 0)
        current_position = robot_positions[-1] if robot_positions else (0, 0)

    robot_positions.append((current_position, time.time()))
    return frame


# Function to generate frames for video streaming
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_robot(frame)

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def team_input():
    return render_template("team_input.html")


@app.route("/start_timer_page", methods=["POST"])
def start_timer_page():
    global team_name
    team_name = request.form.get("team_name")
    return render_template("timer.html", team_name=team_name)


@app.route("/start_system", methods=["POST"])
def start_system():
    global start_time, timer_running
    start_time = time.time()
    timer_running = True
    return jsonify(status="success")


@app.route("/stop_system", methods=["POST"])
def stop_system():
    global timer_running, start_time
    timer_running = False
    elapsed_time = time.time() - start_time if start_time else 0.0
    return jsonify(status="stopped", time_taken=f"{elapsed_time:.2f} seconds")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
