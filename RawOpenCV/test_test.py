import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

typed_text = ""
last_detection_time = 0
detection_cooldown = 1.0
typing_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Much tighter skin detection
    lower_skin = np.array([0, 30, 80], dtype=np.uint8)  # Increased minimums
    upper_skin = np.array([20, 200, 255], dtype=np.uint8)  # Decreased max saturation
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Add noise reduction
    kernel = np.ones((3,3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Much higher minimum area
    large_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    
    # Show the mask for debugging
    cv2.imshow('Skin Mask', skin_mask)
    
    if large_contours:
        largest = max(large_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        cv2.putText(frame, f"Area: {int(area)}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)
        
        cv2.putText(frame, "T", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No hands detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Hand Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()