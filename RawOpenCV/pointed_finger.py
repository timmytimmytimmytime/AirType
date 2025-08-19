import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def detect_pointing_finger(contour):
   """Simple pointing detection based on contour shape"""
   # Get the convex hull
   hull = cv2.convexHull(contour, returnPoints=False)
   
   if len(hull) > 3:
       defects = cv2.convexityDefects(contour, hull)
       if defects is not None:
           # Count fingers by counting defects
           finger_count = 0
           for i in range(defects.shape[0]):
               s, e, f, d = defects[i, 0]
               start = tuple(contour[s][0])
               end = tuple(contour[e][0])
               far = tuple(contour[f][0])
               
               # Calculate angle
               a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
               b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
               c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
               angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
               
               if angle <= np.pi/2:  # 90 degrees
                   finger_count += 1
           
           return finger_count == 1  # Only index finger extended
   return False

while True:
   ret, frame = cap.read()
   if not ret:
       break
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   lower_skin = np.array([0, 15, 50], dtype=np.uint8)
   upper_skin = np.array([25, 255, 255], dtype=np.uint8)
   
   skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
   contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   pointing_detected = False
   
   if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       if cv2.contourArea(largest_contour) > 3000:
           cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
           
           if detect_pointing_finger(largest_contour):
               pointing_detected = True
   
   # Display letter when pointing
   if pointing_detected:
       cv2.putText(frame, "A", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
   
   cv2.imshow('Pointing Detection', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()