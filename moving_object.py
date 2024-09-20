import cv2
import numpy as np


def main():
    # Start capturing video from the webcam or video file
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide path to video
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Failed to capture video.")
        cap.release()
        return

    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create HSV image for visualization of flow (optional)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    while cap.isOpened():
        # Capture the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Convert flow to polar coordinates to visualize
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for visualization
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display the flow visualization
        cv2.imshow("Optical Flow", flow_bgr)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Update the previous frame
        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
