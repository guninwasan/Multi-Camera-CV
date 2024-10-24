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
distance_covered = 0  # Distance in meters
prev_position = None  # Previous robot position
pixel_to_meter_ratio = 0.01  # Example ratio: 1 pixel = 0.01 meters (You need to calibrate this for your system)

aruco_detector = ArucoDetector()
optical_flow = OpticalFlow()
robot_positions = []  # Store robot positions

# Directory to store videos and reports
video_save_directory = os.path.join(os.getcwd(), "videos")
report_directory = os.path.join(os.getcwd(), "reports")
if not os.path.exists(video_save_directory):
    os.makedirs(video_save_directory)
if not os.path.exists(report_directory):
    os.makedirs(report_directory)


# Function to calculate distance between two points (in pixels)
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to calculate robot speed in meters per second
def calculate_speed(current_position, prev_position, elapsed_time):
    # Calculate pixel distance
    pixel_distance = calculate_distance(current_position, prev_position)

    # Convert to meters using pixel-to-meter ratio
    distance_in_meters = pixel_distance * pixel_to_meter_ratio

    # Speed = distance / time
    speed = distance_in_meters / elapsed_time if elapsed_time > 0 else 0
    return speed


# Function to process the frame for robot detection (for display purposes only)
def detect_robot_for_display(frame):
    global robot_positions, prev_position, distance_covered, start_time

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

        # Calculate the speed if tracking started
        if timer_running and prev_position is not None:
            elapsed_time = time.time() - start_time
            speed = calculate_speed(current_position, prev_position, elapsed_time)
            distance_covered += (
                calculate_distance(current_position, prev_position)
                * pixel_to_meter_ratio
            )
            start_time = time.time()  # Reset start time for the next frame

            # Display speed in meters/second
            cv2.putText(
                frame,
                f"Speed: {speed:.2f} m/s",
                (x_center + 10, y_center + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        prev_position = current_position  # Update the previous position

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


# Function to save the team report
def save_team_report(team_name, time_taken, distance_covered, avg_speed):
    report_file = os.path.join(report_directory, f"{team_name}.txt")
    with open(report_file, "w") as report:
        report.write(f"Team Name: {team_name}\n")
        report.write(f"Time Taken: {time_taken:.2f} seconds\n")
        report.write(f"Distance Covered: {distance_covered:.2f} meters\n")
        report.write(f"Average Speed: {avg_speed:.2f} m/s\n")
        report.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


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
    global start_time, timer_running, recording, out, distance_covered, prev_position

    # Start the timer
    start_time = time.time()
    timer_running = True
    distance_covered = 0  # Reset distance
    prev_position = None  # Reset previous position

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
    global timer_running, recording, out, team_name, distance_covered

    # Stop the timer
    elapsed_time = time.time() - start_time if start_time else 0.0
    timer_running = False

    # Stop video recording
    if recording:
        recording = False
        if out is not None:
            out.release()
            out = None

    # Calculate average speed
    avg_speed = distance_covered / elapsed_time if elapsed_time > 0 else 0

    # Save the report for the team
    save_team_report(team_name, elapsed_time, distance_covered, avg_speed)

    return jsonify(status="stopped", time_taken=f"{elapsed_time:.2f} seconds")


# Route to stream the video feed with robot detection
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
