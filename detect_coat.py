import cv2
import numpy as np

def detect_green_coat(frame):
    """
    Detects if a green-colored coat is visible in the lower center of the frame.
    Returns True if green is detected, else False.
    """

    # Resize for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Focus on the torso region
    height, width, _ = resized_frame.shape
    roi = resized_frame[int(height * 0.5):int(height * 0.9), int(width * 0.3):int(width * 0.7)]

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define green range in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Count green pixels
    green_ratio = cv2.countNonZero(mask) / mask.size

    # Return True if green occupies significant region (5%)
    return green_ratio > 0.05
