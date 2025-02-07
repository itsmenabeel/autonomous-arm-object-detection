import torch
import numpy as np
import cv2
from time import time
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class ShapeDetection:

   def __init__(self, capture_index, model_name):
        
      self.capture_index = capture_index                                                                  # Specifies where the input is coming from (webcam, mp4 or other)
      self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, source='github', force_reload=True)     # Using custom model 
      self.classes = self.model.names
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                                        # Uses CUDA GPU acceleration if available
      print("Using Device: ", self.device)           


   def score_frame(self, frame):                                                                          # This function takes each frame and scores it using the custom YOLOv5 model

      self.model.to(self.device)
      frame = [frame]
      results = self.model(frame)
      labels, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
      return labels, coord
   
   
   def calculate_distance(self, center_point, screen_center):
      
      return np.linalg.norm(np.array(center_point) - np.array(screen_center))
   
   
   def detect_color(self, roi):
      
      # Convert the ROI to the HSV color space
      hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

      # Define color ranges for red, yellow, and blue
      lower_red = np.array([0, 50, 70])
      upper_red = np.array([9, 255, 255])
      lower_yellow = np.array([20, 50, 70])
      upper_yellow = np.array([35, 255, 255])
      lower_blue = np.array([90, 50, 70])
      upper_blue = np.array([128, 255, 255])
      
      mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
      mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
      mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
      
      kernel = np.ones((9, 9), np.uint8)  
      mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
      mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
      mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
      
      contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Check if the average color falls within a certain threshold for each color
      for contours, color, label in zip([contours_red, contours_yellow, contours_blue], [(0, 0, 255), (0, 255, 255), (255, 0, 0)], ["Prism", "Cylinder", "Cube"]):
        for _ in contours:
           return color, label  


   def plot_boxes(self, results, frame):
      
      labels, coord = results
      n = len(labels)
      x_shape, y_shape = frame.shape[1], frame.shape[0]
      screen_center = (x_shape // 2, y_shape // 2)
      
      for i in range(n):
         row = coord[i]

         if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

            # Extract the region of interest (ROI) for color detection
            roi = frame[y1:y2, x1:x2]

            # Perform color detection on the ROI
            color, label = self.detect_color(roi)

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_point = (center_x, center_y)

            # Calculate the distance from the center of the monitor
            distance = self.calculate_distance(center_point, screen_center)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put text on the frame with the new label and distance
            cv2.putText(frame, f"{label} - {distance:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

      return frame


   def run(self):

      cap = cv2.VideoCapture(self.capture_index)                                                         # Gets the feed
      
      while True:
         _, frame = cap.read()
            
         frame = cv2.resize(frame, (1280, 720))                                                            # Resizing the frame to 640, since the model was trained on that
            
         start_time = time()
         results = self.score_frame(frame)
         frame = self.plot_boxes(results, frame)  
         end_time = time()
         
         fps = 1/np.round(end_time - start_time, 2)
             
         cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
         cv2.imshow('3D Objects', frame)
 
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
      cap.release()
      cv2.destroyAllWindows()
        

# Driver Code
detector = ShapeDetection(capture_index = 'object_vid_1.mp4', model_name = 'best.pt')
detector.run()