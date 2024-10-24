# Multi Camera CV System

## Description

This system is designed to capture and process video feeds from three different webcams, displaying them as a combined single feed for easier robot tracking. The system uses ArUco marker detection and optical flow to track a robot’s position and calculate its speed. The captured data, including the robot's speed and distance covered, is stored in a report for each team.

Key features include:

- Simultaneous capture and display of three webcam feeds.
- ArUco marker detection and optical flow for robot tracking.
- Real-time robot position and speed calculation.
- Recording of raw video from all three webcams combined into a single video file.
- Generation of team-specific reports with time taken, distance covered, and average speed.

## Pre-requisites

To run the system, ensure you have the following:

1. **Python 3.7+**
2. Install all required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Webcams**: Make sure three webcams are connected to your system and recognized by OpenCV.

## Running the system

- Clone the repository (or download the project files).
- Run the Flask application by executing the following command in your terminal: `python3 app.py`.
- Open your browser and enter the team name
- Click the `Start` button to begin tracking and recording.
- Starting the time also starts recording the video feed.
- To stop the system, click the `Stop` button. A report will be generated and saved in the `reports/` directory along with the recorded video in the `videos/` directory.

The system will automatically resize the video feeds from all webcams to the same size (640x480) for proper display and recording.
