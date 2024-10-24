import cv2
from flask import Flask, render_template, request, jsonify, Response
import time
import os
import threading
import numpy as np
from aruco import ArucoDetector
from moving_object import OpticalFlow

app = Flask(__name__)

# Global variables
cap = cv2.VideoCapture(0)  # Capture video feed from the default camera
team_name = None
timer_running = False
start_time = None
lock = threading.Lock()
recording = False  # To track recording status
out = None  # VideoWriter object

aruco_detector = ArucoDetector()
optical_flow = OpticalFlow()
robot_positions = []  # Store robot positions

# Directory to store videos
video_save_directory = os.path.join(os.getcwd(), "videos")
if not os.path.exists(video_save_directory):
    os.makedirs(video_save_directory)


# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to process the frame for robot detection (for display purposes only)
def detect_robot_for_display(frame):
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
    global recording, out, cap

    while True:
        success, frame = cap.read()
        if not success:
            break

        raw_frame = frame.copy()  # Capture the raw frame for recording

        # If recording, write the raw frame (without ArUco/Optical flow)
        if recording and out is not None:
            out.write(raw_frame)

        # For display purposes, process the frame with ArUco markers and optical flow
        display_frame = detect_robot_for_display(frame)

        # Encode frame for display
        ret, buffer = cv2.imencode(".jpg", display_frame)
        display_frame = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + display_frame + b"\r\n"
        )


# Route to get team input
@app.route("/")
def team_input():
    return render_template("team_input.html")


# Route to start the timer page
@app.route("/start_timer_page", methods=["POST"])
def start_timer_page():
    global team_name
    team_name = request.form.get("team_name")
    return render_template("timer.html", team_name=team_name)


# Route to start the system (start timer and recording)
@app.route("/start_system", methods=["POST"])
def start_system():
    global start_time, timer_running, recording, out

    # Start the timer
    start_time = time.time()
    timer_running = True

    # Start video recording if not already recording
    if not recording:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        out = cv2.VideoWriter(
            os.path.join(video_save_directory, f"{team_name}_{int(start_time)}.mp4"),
            fourcc,
            20.0,
            (frame_width, frame_height),
        )
        recording = True

    return jsonify(status="success")


# Route to stop the system (stop timer and recording)
@app.route("/stop_system", methods=["POST"])
def stop_system():
    global timer_running, recording, out

    # Stop the timer
    timer_running = False
    elapsed_time = time.time() - start_time if start_time else 0.0

    # Stop video recording
    if recording:
        recording = False
        if out is not None:
            out.release()
            out = None

    return jsonify(status="stopped", time_taken=f"{elapsed_time:.2f} seconds")


# Route to stream the video feed with robot detection
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
