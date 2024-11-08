import cv2
from flask import Flask, render_template, request, jsonify, Response
import time
import os
import threading
import numpy as np
from aruco import ArucoDetector
from moving_object import OpticalFlow
import atexit


app = Flask(__name__)

# Global variables for multiple cameras
cap1 = cv2.VideoCapture(0)  # First webcam
cap2 = cv2.VideoCapture(1)  # Second webcam
cap3 = cv2.VideoCapture(2)  # Third webcam
team_name = None
timer_running = False
start_time = None
lock = threading.Lock()
recording = False  # To track recording status
out = None  # VideoWriter object
distance_covered = 0
prev_position = None
pixel_to_meter_ratio = 1

aruco_detector = ArucoDetector()
optical_flow_on = True
optical_flow = OpticalFlow()
robot_positions = []

# Directory to store videos and reports
video_save_directory = os.path.join(os.getcwd(), "videos")
report_directory = os.path.join(os.getcwd(), "reports")
if not os.path.exists(video_save_directory):
    os.makedirs(video_save_directory)
if not os.path.exists(report_directory):
    os.makedirs(report_directory)


def release_resources():
    global cap1, cap2, cap3
    if cap1.isOpened():
        cap1.release()
    if cap2.isOpened():
        cap2.release()
    if cap3.isOpened():
        cap3.release()
    print("Cameras released successfully.")


atexit.register(release_resources)


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
            # cv2.putText(
            #     frame,
            #     f"Speed: {speed:.2f} px/s",
            #     (x_center + 10, y_center + 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.9,
            #     (255, 0, 0),
            #     2,
            # )

        prev_position = current_position  # Update the previous position

    elif optical_flow_on:
        # Optical flow when no marker is detected
        flow_bgr = optical_flow.calculate_flow(frame)
        alpha = 0.5  # Transparency factor
        frame = cv2.addWeighted(flow_bgr, alpha, frame, 1 - alpha, 0)
        current_position = robot_positions[-1] if robot_positions else (0, 0)
    else:
        current_position = (0, 0)

    robot_positions.append((current_position, time.time()))
    return frame


# Function to stack frames horizontally or vertically
def stack_frames(frames, axis=1):
    return np.concatenate(frames, axis=axis)


# Function to resize frames to a consistent size
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))


# Function to save the team report
def save_team_report(team_name, time_taken, distance_covered, avg_speed):
    report_file = os.path.join(report_directory, f"{team_name}.txt")
    with open(report_file, "a") as report:
        report.write(f"Team Name: {team_name}\n")
        report.write(f"Time Taken: {time_taken:.2f} seconds\n")
        report.write(f"Distance Covered: {distance_covered:.2f} meters\n")
        report.write(f"Average Speed: {avg_speed:.2f} p/s\n")
        report.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write("\n\n")


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
        target_width = 640  # Assuming the resized frame width
        target_height = 480  # Assuming the resized frame height

        # Calculate the width for combined frame
        combined_frame_width = target_width * 3  # 3 cameras horizontally
        combined_frame_height = target_height  # Height remains the same

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        out = cv2.VideoWriter(
            os.path.join(video_save_directory, f"{team_name}_{int(start_time)}.mp4"),
            fourcc,
            20.0,  # FPS
            (combined_frame_width, combined_frame_height),  # Combined frame size
        )
        recording = True

    return jsonify(status="success")


# Function to generate frames for video streaming and recording
def generate_frames():
    global recording, out, cap1, cap2, cap3

    target_width = 640
    target_height = 480

    while True:
        # Read from each camera
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        success3, frame3 = cap3.read()

        if not success1 or not success2 or not success3:
            break

        # Resize all frames to the target size
        frame1_resized = resize_frame(frame1, target_width, target_height)
        frame2_resized = resize_frame(frame2, target_width, target_height)
        frame3_resized = resize_frame(frame3, target_width, target_height)

        # Combine the resized frames horizontally (can change to vertical by using axis=0)
        combined_frame = stack_frames(
            [frame1_resized, frame2_resized, frame3_resized], axis=1
        )

        # Process frames (optional: e.g., ArUco or optical flow can be applied here)
        display_frame1 = detect_robot_for_display(frame1_resized)
        display_frame2 = detect_robot_for_display(frame2_resized)
        display_frame3 = detect_robot_for_display(frame3_resized)

        # Combine processed frames for display
        combined_display_frame = stack_frames(
            [display_frame1, display_frame2, display_frame3], axis=1
        )

        # If recording, write the combined raw frame (without ArUco/Optical flow)
        if recording and out is not None:
            try:
                # Write the combined frame to the video file
                out.write(combined_frame)
            except Exception as e:
                print(f"Error writing to video file: {e}")

        # Encode combined display frame for streaming
        ret, buffer = cv2.imencode(".jpg", combined_display_frame)
        combined_display_frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + combined_display_frame + b"\r\n"
        )


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
