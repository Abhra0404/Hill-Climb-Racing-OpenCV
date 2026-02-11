import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands with new API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Download model if not exists
import os
if not os.path.exists('hand_landmarker.task'):
    import urllib.request
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'hand_landmarker.task'
    )
    print("Model downloaded!")

def is_fist(hand_landmarks):
    """Detect if hand is closed (fist) by checking finger curl."""
    # Get palm center (wrist landmark 0)
    palm = hand_landmarks[0]
    
    # Finger tip landmarks: thumb=4, index=8, middle=12, ring=16, pinky=20
    # Finger middle landmarks: thumb=2, index=6, middle=10, ring=14, pinky=18
    fingertips = [4, 8, 12, 16, 20]
    finger_mids = [2, 6, 10, 14, 18]
    
    # Count fingers that are curled (tip closer to palm than middle joint)
    curled_count = 0
    for tip_idx, mid_idx in zip(fingertips, finger_mids):
        tip = hand_landmarks[tip_idx]
        mid = hand_landmarks[mid_idx]
        
        # Distance from tip to palm vs mid to palm
        tip_dist = ((tip.x - palm.x)**2 + (tip.y - palm.y)**2)**0.5
        mid_dist = ((mid.x - palm.x)**2 + (mid.y - palm.y)**2)**0.5
        
        if tip_dist < mid_dist * 1.1:  # Tip is closer = finger curled
            curled_count += 1
    
    # Fist if at least 4 fingers are curled
    return curled_count >= 4

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

screen_width = 640
left_key_pressed = False  # Track if left key is held
right_key_pressed = False  # Track if right key is held

print("🎮 Hill Climb Gesture Control - DUAL HAND MODE")
print("LEFT HAND:  ✊ Close fist → HOLD LEFT pedal")
print("RIGHT HAND: ✊ Close fist → HOLD RIGHT pedal")
print("✋ Open hand → RELEASE pedal")
print("Use BOTH hands together to accelerate! 🚗\n")

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Detect hands
    results = detector.detect(mp_image)
    num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
    
    left_hand_fist = False
    right_hand_fist = False
    
    if num_hands >= 1:
        # Process each detected hand
        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Determine if this is left or right hand based on x position
            wrist_x = hand_landmarks[0].x
            is_left_hand = wrist_x < 0.5  # Left side of frame = left hand
            
            # Check if fist is closed
            fist_closed = is_fist(hand_landmarks)
            
            if is_left_hand:
                left_hand_fist = fist_closed
                color = (0, 255, 0) if fist_closed else (100, 100, 100)
            else:
                right_hand_fist = fist_closed
                color = (0, 255, 0) if fist_closed else (100, 100, 100)
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x_px = int(landmark.x * img.shape[1])
                y_px = int(landmark.y * img.shape[0])
                cv2.circle(img, (x_px, y_px), 4, color, -1)
        
        # Control left hand independently
        if left_hand_fist:
            if not left_key_pressed:
                pyautogui.keyDown("left")
                left_key_pressed = True
                print("✊ LEFT HAND FIST - HOLDING LEFT")
        else:
            if left_key_pressed:
                pyautogui.keyUp("left")
                left_key_pressed = False
                print("✋ LEFT HAND OPEN - RELEASED LEFT")
        
        # Control right hand independently
        if right_hand_fist:
            if not right_key_pressed:
                pyautogui.keyDown("right")
                right_key_pressed = True
                print("✊ RIGHT HAND FIST - HOLDING RIGHT")
        else:
            if right_key_pressed:
                pyautogui.keyUp("right")
                right_key_pressed = False
                print("✋ RIGHT HAND OPEN - RELEASED RIGHT")
        
        # Display status
        if left_hand_fist and right_hand_fist:
            cv2.putText(img, "🚗 BOTH PEDALS!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif left_hand_fist:
            cv2.putText(img, "LEFT FIST", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif right_hand_fist:
            cv2.putText(img, "RIGHT FIST", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(img, "HANDS OPEN", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # No hands detected
    else:
        # Release both keys
        if left_key_pressed:
            pyautogui.keyUp("left")
            left_key_pressed = False
        if right_key_pressed:
            pyautogui.keyUp("right")
            right_key_pressed = False

    # Show direction zones
    third = img.shape[1] // 3
    cv2.line(img, (third, 0), (third, img.shape[0]), (255, 0, 0), 2)
    cv2.line(img, (third*2, 0), (third*2, img.shape[0]), (255, 0, 0), 2)
    cv2.putText(img, "LEFT", (20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "RIGHT", (third*2 + 20, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up - release any held keys
if left_key_pressed:
    pyautogui.keyUp("left")
if right_key_pressed:
    pyautogui.keyUp("right")

cap.release()
cv2.destroyAllWindows()
