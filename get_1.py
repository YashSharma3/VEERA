import mediapipe as mp
import cv2
import pygame
import math
import requests  # Import requests for API calls

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Pygame mixer
#pygame.mixer.init()
#pygame.mixer.music.load('Emergency.mp3')  # Replace with the path to your sound file

# API endpoint and data for SOS request
api_url = 'http://13.126.70.254:8000/dashboard/api/sos-call/'  # Your API endpoint
api_data = {
    "message": "M samasya m hu",
    "name": "DK",
    "phone_number": "9898989898",
    "email": "divyanshukhandelwal098@gmail.com",
    "latitude": 23.87,
    "longitude": 34.56
}

# Initialize webcam.
cap = cv2.VideoCapture(0)

# Gesture counter for crossed hands
crossed_hand_count = 0
current_gesture = "None"
gesture_detected = False  # To ensure the count only increases once per detection

# Calculate Euclidean distance between two landmarks
def euclidean_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) * 100

# Detect if hands are crossed and return the distance between them
def are_hands_crossed(left_hand_landmarks, right_hand_landmarks):
    # Get key landmarks for each hand
    left_wrist = left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    right_wrist = right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Normalize based on distance between wrist and index finger tip on left hand
    left_index_finger_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    normalization_distance = euclidean_distance(left_wrist, left_index_finger_tip)

    # Calculate the distance between the left and right wrist
    wrist_distance = euclidean_distance(left_wrist, right_wrist) / normalization_distance
    print(f"Normalized distance between hands (wrist to wrist): {wrist_distance:.4f}")  # Print the normalized distance

    # Threshold to determine if hands are crossed (adjust based on testing)
    threshold = 1.0  # Adjust this value based on testing

    # If the normalized distance between the wrists is below the threshold, hands are crossed
    return wrist_distance < threshold

# Start hand detection.
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
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
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            # Get hand landmarks for both hands
            left_hand_landmarks = results.multi_hand_landmarks[0]
            right_hand_landmarks = results.multi_hand_landmarks[1]

            # Check if hands are crossed and print distance
            if are_hands_crossed(left_hand_landmarks, right_hand_landmarks):
                if not gesture_detected:
                    current_gesture = "Crossed"
                    crossed_hand_count += 1
                    gesture_detected = True  # Mark that the gesture has been detected
                    print(f"Crossed hands detected {crossed_hand_count} times.")

                    # Play sound and reset the count after every 3 gestures
                    if crossed_hand_count >= 3:
                        print("******EMERGENCY****")
                        #pygame.mixer.music.play()  # Play the emergency sound

                        # Send SOS API request
                        try:
                            response = requests.post(api_url, json=api_data)  # Send API request
                            print(f"SOS call sent. Server response: {response.status_code}")
                        except Exception as e:
                            print(f"Failed to send SOS call: {e}")

                        crossed_hand_count = 0  # Reset the counter after emergency call

            else:
                gesture_detected = False  # Reset the flag when hands are no longer crossed

            # Draw landmarks for both hands
            mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

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
