import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def detect_prayer_gesture(contours):
    if not contours:
        return False
    
    # Only consider the largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # Very strict requirements
    if area > 12000:  # Much larger minimum
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = h / w
        
        # Must be tall and narrow like prayer hands
        if aspect_ratio > 1.5:
            return True
    
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Much tighter skin detection to reduce noise
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 40, 100], dtype=np.uint8)  # Even stricter
    upper_skin = np.array([15, 180, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Heavy noise reduction
    kernel = np.ones((7,7), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 8000]
    
    if detect_prayer_gesture(large_contours):
        cv2.putText(frame, "PRAYER DETECTED!", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    
    if large_contours:
        largest = max(large_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        ratio = h / w if w > 0 else 0
        cv2.putText(frame, f"Area: {int(area)}, Ratio: {ratio:.2f}", 
                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.drawContours(frame, [largest], -1, (0, 255, 255), 2)
    
    cv2.imshow('Prayer Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()