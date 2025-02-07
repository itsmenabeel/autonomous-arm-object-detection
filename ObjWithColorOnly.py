import cv2
import numpy as np
import serial as ser
import time

#arduino = ser.Serial("COM8", 115200)
# Calculate the distance from the center of the monitor
def calculate_distance(center_point, screen_center):
    return np.linalg.norm(np.array(center_point) - np.array(screen_center))

def send_to_arduino(board, data):
    board.write(data.encode())
    time.sleep(0.1)

# Capture video from camera
cap = cv2.VideoCapture(0)

# Get screen resolution
screen_width, screen_height = 1280, 720
screen_center = (screen_width // 2, screen_height // 2)

# Create a window to display the video feed
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.convertScaleAbs(frame, alpha = 1.4, beta = 40)

    # Convert the frame to HSV for color-based segmentation
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and blue
    lower_red = np.array([0, 50, 70])
    upper_red = np.array([9, 255, 255])
    lower_yellow = np.array([20, 50, 70])
    upper_yellow = np.array([35, 255, 255])
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])

    # Threshold the HSV frame to get binary masks for each color
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    kernel = np.ones((9, 9), np.uint8)  
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    # Find contours in the masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours for each color
    for contours, color, label in zip([contours_red, contours_yellow, contours_blue], [(0, 0, 255), (0, 255, 255), (255, 0, 0)], ["Prism", "Cylinder", "Cube"]):
        for contour in contours:
            # Find the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            center_x_cm, center_y_cm = center_x // 37.79, center_y // 37.79
            center_point = (center_x_cm, center_y_cm)
            #send_to_arduino(arduino, f"{center_x},{center_y}\n")

            distance = calculate_distance(center_point, screen_center)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, f"{label} - Center: {center_point}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
