import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# Text persistence variables
typed_text = ""
last_detection_time = 0
last_detected_letter = None
detection_cooldown = 1.0  # 1 second buffer

def get_fingertip_positions(contour):
   """Find fingertip positions using convex hull"""
   hull = cv2.convexHull(contour, returnPoints=True)
   
   fingertips = []
   for point in hull:
       x, y = point[0]
       fingertips.append((x, y))
   
   # Sort by y-coordinate (topmost first) and take top 4-5 points
   fingertips = sorted(fingertips, key=lambda p: p[1])[:5]
   # Sort by x-coordinate (left to right)
   fingertips = sorted(fingertips, key=lambda p: p[0])
   
   return fingertips

def detect_extended_finger(fingertips):
   """Determine which finger is most extended (highest point)"""
   if len(fingertips) < 2:
       return None
   
   # Find the highest fingertip
   highest = min(fingertips, key=lambda p: p[1])
   
   # Map position to finger (left to right: pinky, ring, middle, index)
   finger_map = {0: 'A', 1: 'S', 2: 'D', 3: 'F'}
   
   # Find index of highest point
   sorted_tips = sorted(fingertips, key=lambda p: p[0])
   if highest in sorted_tips:
       finger_index = sorted_tips.index(highest)
       return finger_map.get(finger_index, None)
   
   return None

while True:
   ret, frame = cap.read()
   if not ret:
       break
   
   # Mirror the image
   frame = cv2.flip(frame, 1)
   
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   lower_skin = np.array([0, 15, 50], dtype=np.uint8)
   upper_skin = np.array([25, 255, 255], dtype=np.uint8)
   
   skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
   contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   detected_letter = None
   current_time = time.time()
   
   if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       if cv2.contourArea(largest_contour) > 3000:
           cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
           
           fingertips = get_fingertip_positions(largest_contour)
           detected_letter = detect_extended_finger(fingertips)
           
           # Draw fingertip points
           for tip in fingertips:
               cv2.circle(frame, tip, 8, (255, 0, 0), -1)
   
   # Add letter to text with cooldown (prevent repeated letters)
   if (detected_letter and 
       detected_letter != last_detected_letter and 
       (current_time - last_detection_time) > detection_cooldown):
       typed_text += detected_letter
       last_detected_letter = detected_letter
       last_detection_time = current_time
   elif not detected_letter:
       last_detected_letter = None
   
   # Display accumulated text
   cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
   
   # Show current detection
   if detected_letter:
       cv2.putText(frame, f"Detecting: {detected_letter}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
   
   cv2.imshow('Finger Typing', frame)
   
   key = cv2.waitKey(1) & 0xFF
   if key == ord('q'):
       break
   elif key == ord(' '):  # Space bar to add space
       typed_text += " "
   elif key == 8:  # Backspace
       typed_text = typed_text[:-1]

cap.release()
cv2.destroyAllWindows()

print(f"Final text: {typed_text}")