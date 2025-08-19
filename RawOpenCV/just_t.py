import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

typed_text = ""
last_detection_time = 0
detection_cooldown = 1.0

def create_background_subtractor():
    """Create background subtractor to focus on moving objects"""
    return cv2.createBackgroundSubtractorMOG2(detectShadows=False)

bg_subtractor = create_background_subtractor()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Background subtraction to focus on moving objects
    fg_mask = bg_subtractor.apply(frame)
    
    # Combine with skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 40, 100], dtype=np.uint8)
    upper_skin = np.array([15, 180, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Only consider skin-colored moving objects
    combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
    
    # Clean up
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    
    # Look for moving skin-colored objects
    large_contours = [c for c in contours if cv2.contourArea(c) > 2000]
    
    if (large_contours and (current_time - last_detection_time) > detection_cooldown):
        typed_text += "T"
        last_detection_time = current_time
    
    # Debug visualization
    cv2.imshow('Moving Skin Objects', combined_mask)
    if large_contours:
        cv2.drawContours(frame, large_contours, -1, (0, 255, 0), 2)
    
    cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow('Hand Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        typed_text = ""

cap.release()
cv2.destroyAllWindows()