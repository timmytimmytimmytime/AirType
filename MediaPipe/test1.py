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

def detect_left_index_finger(hand_landmarks, handedness):
    """Detect if left hand index finger is extended"""
    if handedness.classification[0].label != "Left":
        return False
    
    # Index finger landmarks: 5 (base), 6, 7, 8 (tip)
    index_tip = hand_landmarks.landmark[8]
    index_mcp = hand_landmarks.landmark[5]  # Base joint
    
    # Check if index finger is extended (tip higher than base)
    if index_tip.y < index_mcp.y - 0.05:  # Pointing up
        return True
    
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    current_time = time.time()
    detected = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check for left index finger
            if detect_left_index_finger(hand_landmarks, handedness):
                detected = True
                cv2.putText(frame, "LEFT INDEX DETECTED", (300, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add T with cooldown
    if detected and (current_time - last_detection_time) > detection_cooldown:
        typed_text += "T"
        last_detection_time = current_time
    
    # Display accumulated text
    cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow('MediaPipe Hand Tracking', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        typed_text = ""

cap.release()
cv2.destroyAllWindows()
print(f"Final text: {typed_text}")