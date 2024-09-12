import cv2
import torch
import os

# Set the path to the YOLOv5 local directory
yolo_dir = os.path.join(os.getcwd(), "yolov5")

# Load the YOLOv5 model from local directory
model = torch.hub.load(yolo_dir, "custom", path="yolov5s.pt", source="local")


# Define function to perform object detection
def detect_objects(frame):
    results = model(frame)  # Run inference on the frame
    return results


# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Perform object detection
    results = detect_objects(frame)

    # Render the detection results on the frame
    results.render()  # Updates the frame with bounding boxes and labels

    # Display the frame with detections
    cv2.imshow("YOLOv5 Real-time Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
