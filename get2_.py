import mediapipe as mp
import cv2
import requests  # Import the requests library
import pygame

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Pygame mixer
#pygame.mixer.init()
#pygame.mixer.music.load('Emergency.mp3')  # Replace with the path to your sound file

# Initialize webcam.
cap = cv2.VideoCapture(0)

# Gesture counters for open and closed hands
gesture_counts = {'open': 0, 'closed': 0}
current_gesture = "None"

# Function to send SOS POST request
def send_sos_call():
    url = 'http://13.126.70.254:8000/dashboard/api/sos-call/'  # API endpoint
    data = {
        "message": "M samasya m hu",
        "name": "DK",
        "phone_number": "9898989898",
        "email": "divyanshukhandelwal098@gmail.com",
        "latitude": 23.87,
        "longitude": 34.56
    }

    try:
        res = requests.post(url, json=data)  # Sending the POST request with the message
        print(f'SOS Call Sent. Server Response: {res.status_code}')
    except Exception as e:
        print(f'Error sending SOS call: {e}')

# To store the current gesture (Open or Closed)
def is_open_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x

    hand_width = abs(index_tip - pinky_tip)

    if abs(index_tip - thumb_tip) / hand_width > 0.43:
        return True
    return False

def is_closed_hand(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
    return abs(index_tip - pinky_tip) < 0.1

# Start hand detection.
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            # Process only the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            if handedness == "Right":
                # Draw landmarks for the right hand only
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                # Display X-axis coordinates for each landmark
                h, w, _ = image.shape
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    x_coord = f'x: {landmark.x:.2f}'
                    cv2.putText(image, x_coord, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

                # Detect open or closed hand
                if is_open_hand(hand_landmarks):
                    if current_gesture != "Open":
                        current_gesture = "Open"
                        gesture_counts['open'] += 1
                        print(f"Open hand detected {gesture_counts['open']} times.")

                elif is_closed_hand(hand_landmarks):
                    if current_gesture != "Closed":
                        current_gesture = "Closed"
                        gesture_counts['closed'] += 1
                        print(f"Closed hand detected {gesture_counts['closed']} times.")

            if gesture_counts['closed'] >= 5:
                print("***************EMERGENCY*************")
                #pygame.mixer.music.play()  # Play the emergency sound
                send_sos_call()  # Send the SOS POST request
                gesture_counts['closed'] = 0
                gesture_counts['open'] = 0

        # Display the current gesture on the video feed
        cv2.putText(image, f'Current Gesture: {current_gesture}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
