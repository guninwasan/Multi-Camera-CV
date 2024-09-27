from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import threading
from aruco import ArucoDetector
from moving_object import OpticalFlow
from yolo import YoloDetector

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global variables
cap = cv2.VideoCapture(0)  # Change this to your video source if needed
start_time = None
timer_running = False
human_interventions = 0
team_name = "Team"

# Lock for thread-safe operations
lock = threading.Lock()

# Initialize detectors
aruco_detector = ArucoDetector()
optical_flow = OpticalFlow()
yolo_detector = YoloDetector()


def generate_frames():
    global start_time, timer_running, human_interventions

    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            # Aruco tag detection
            corners, ids = aruco_detector.detect_markers(frame)
            if ids is not None:
                for i, corner in enumerate(corners):
                    c = corner[0]
                    x_center = int(c[:, 0].mean())
                    y_center = int(c[:, 1].mean())
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

            # Optical flow
            flow_bgr = optical_flow.calculate_flow(frame)
            alpha = 0.5  # Transparency factor
            frame = cv2.addWeighted(flow_bgr, alpha, frame, 1 - alpha, 0)

            # YOLO detection
            results = yolo_detector.detect_objects(frame)
            for result in results.xyxy[0]:
                if result[-1] == 0:  # Assuming class 0 is 'person'
                    human_interventions += 1
                    break

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
    global start_time, timer_running, human_interventions
    with lock:
        start_time = None
        timer_running = False
        human_interventions = 0
    return jsonify(status="success")


@app.route("/update_team", methods=["POST"])
def update_team():
    global team_name
    team_name = request.form["team_name"]
    return jsonify(status="success")


@app.route("/scorecard")
def scorecard():
    global start_time, timer_running, human_interventions
    elapsed_time = 0.00
    if timer_running and start_time is not None:
        elapsed_time = time.time() - start_time
    return jsonify(
        elapsed_time=f"{elapsed_time:.2f} s", human_interventions=human_interventions
    )


if __name__ == "__main__":
    app.run(debug=True)
