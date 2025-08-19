# def check_reset_button_touch(hand_landmarks, handedness, frame_shape):
#     if handedness.classification[0].label != "Right":
#         return False
    
#     index_tip = hand_landmarks.landmark[8]
    
#     # Use actual frame dimensions
#     h, w = frame_shape[:2]
#     x = int(index_tip.x * w)
#     y = int(index_tip.y * h)
    
#     # Debug: draw finger position
#     cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
    
#     button_x, button_y = RESET_BUTTON_POS
#     button_w, button_h = RESET_BUTTON_SIZE
    
#     if (button_x <= x <= button_x + button_w and 
#         button_y <= y <= button_y + button_h):
#         return True
    
#     return False

# # In main loop, call it with:
# if check_reset_button_touch(hand_landmarks, handedness, frame.shape):