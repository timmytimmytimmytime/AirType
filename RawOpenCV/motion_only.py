import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

typed_text = ""
last_detection_time = 0
detection_cooldown = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Motion detection
    fg_mask = bg_subtractor.apply(frame)
    
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    
    # Debug info
    hand_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 15000]
    
    # Show detection status
    cv2.putText(frame, f"Contours found: {len(hand_contours)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if hand_contours:
        largest = max(hand_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        cv2.putText(frame, f"Largest area: {int(area)}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.drawContours(frame, hand_contours, -1, (0, 255, 0), 2)
        
        if (current_time - last_detection_time) > detection_cooldown:
            typed_text += "T"
            last_detection_time = current_time
    
    # Force text display
    cv2.putText(frame, f"Text: {typed_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(frame, "TEST DISPLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Debug Motion', frame)
    cv2.imshow('Motion Mask', fg_mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        typed_text = ""
    elif key == ord('t'):  # Manual T for testing
        typed_text += "T"

cap.release()
cv2.destroyAllWindows()