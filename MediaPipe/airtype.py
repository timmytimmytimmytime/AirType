import cv2
import mediapipe as mp
import time


# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
   static_image_mode=False,
   max_num_hands=2,
   min_detection_confidence=0.7,
   min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

typed_text = ""
last_detection_time = 0
detection_cooldown = 1.0

# Reset button properties
RESET_BUTTON_POS = (500, 50)  # Top right area
RESET_BUTTON_SIZE = (100, 50)
reset_button_active = False

def is_finger_extended(hand_landmarks, finger_tip_id, finger_pip_id):
   """Check if a finger is extended by comparing tip to middle joint"""
   tip = hand_landmarks.landmark[finger_tip_id]
   pip = hand_landmarks.landmark[finger_pip_id]
   return tip.y < pip.y - 0.03

def detect_gesture(hand_landmarks, handedness):
   """Detect specific finger gestures"""
   hand_label = handedness.classification[0].label
   
   if hand_label == "Left":
       index_extended = is_finger_extended(hand_landmarks, 8, 6)
       middle_extended = is_finger_extended(hand_landmarks, 12, 10)
       ring_extended = is_finger_extended(hand_landmarks, 16, 14)
       
       extended_count = sum([index_extended, middle_extended, ring_extended])
       
       if extended_count == 1:
           if index_extended:
               return "T"
           elif middle_extended:
               return "E"
           elif ring_extended:
               return "S"
   
   elif hand_label == "Right":
       thumb_tip = hand_landmarks.landmark[4]
       thumb_mcp = hand_landmarks.landmark[2]
       
       if thumb_tip.y > thumb_mcp.y + 0.05:
           return "SPACE"
   
   return None

def check_reset_button_touch(hand_landmarks, handedness, frame_shape):
    if handedness.classification[0].label != "Right":
        return False
    
    index_tip = hand_landmarks.landmark[8]
    
    # Use actual frame dimensions
    h, w = frame_shape[:2]
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    
    # Debug: draw finger position
    cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
    
    button_x, button_y = RESET_BUTTON_POS
    button_w, button_h = RESET_BUTTON_SIZE
    
    if (button_x <= x <= button_x + button_w and 
        button_y <= y <= button_y + button_h):
        return True
    
    return False


# def check_reset_button_touch(hand_landmarks, handedness):
#    """Check if right hand index finger is touching reset button area"""
#    if handedness.classification[0].label != "Right":
#        return False
   
#    # Get index finger tip position
#    index_tip = hand_landmarks.landmark[8]
   
#    # Convert to pixel coordinates (assuming 640x480 frame)
#    x = int(index_tip.x * 640)
#    y = int(index_tip.y * 480)
   
#    # Check if finger is in reset button area
#    button_x, button_y = RESET_BUTTON_POS
#    button_w, button_h = RESET_BUTTON_SIZE
   
#    if (button_x <= x <= button_x + button_w and 
#        button_y <= y <= button_y + button_h):
#        return True
   
#    return False

while True:
   ret, frame = cap.read()
   if not ret:
       break
   
   frame = cv2.flip(frame, 1)
   rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
   results = hands.process(rgb_frame)
   
   current_time = time.time()
   detected_gesture = None
   reset_button_active = False
   
   if results.multi_hand_landmarks and results.multi_handedness:
       for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
           mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           
           # Check for reset button touch
        #    if check_reset_button_touch(hand_landmarks, handedness):
        #        reset_button_active = True
           if check_reset_button_touch(hand_landmarks, handedness, frame.shape):
               reset_button_active = True
           
           # Detect gesture
           gesture = detect_gesture(hand_landmarks, handedness)
           if gesture:
               detected_gesture = gesture
               display_text = "SPACE" if gesture == "SPACE" else gesture
               cv2.putText(frame, f"Detected: {display_text}", (300, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   
   # Handle reset button
   if reset_button_active and (current_time - last_detection_time) > detection_cooldown:
       typed_text = ""
       last_detection_time = current_time
   
   # Add character with cooldown
   elif detected_gesture and (current_time - last_detection_time) > detection_cooldown:
       if detected_gesture == "SPACE":
           typed_text += " "
       else:
           typed_text += detected_gesture
       last_detection_time = current_time
   
   # Draw reset button
   button_x, button_y = RESET_BUTTON_POS
   button_w, button_h = RESET_BUTTON_SIZE
   button_color = (0, 0, 255) if reset_button_active else (100, 100, 100)
   cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), button_color, -1)
   cv2.putText(frame, "RESET", (button_x + 15, button_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   
   # Display accumulated text
   cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
   
   cv2.imshow('Hand Typing with Reset', frame)
   
   key = cv2.waitKey(1) & 0xFF
   if key == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()
print(f"Final text: {typed_text}")