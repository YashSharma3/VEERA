import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import ssl
import time

# Load the YOLO model for person detection
person_model = YOLO("yolov8n.pt")

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Ensure the camera is opened successfully
assert cap.isOpened(), "Error opening camera"

# Load the pre-trained gender detection model
ssl._create_default_https_context = ssl._create_unverified_context
gender_model = load_model("/Users/yashsharma/PycharmProjects/Veera/Person/gender_model.h5")  # Using Keras model load

# Classes for gender detection
gender_classes = ['Male', 'Female']

# Preprocess function for gender classification model
def preprocess_face(face):
    face = cv2.resize(face, (96, 96))  # Resize face image to the model's expected input size
    face = face.astype("float") / 255.0  # Normalize to [0, 1]
    face = img_to_array(face)  # Convert to array
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Initialize timing for printing every 2 seconds
last_print_time = time.time()

while cap.isOpened():
    # Capture frame-by-frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # Perform inference on the current frame to detect persons (class ID 0)
    results = person_model(frame, classes=[0])  # Detect only persons

    # Initialize gender counts for this frame
    male_count = 0
    female_count = 0

    # Iterate over detected persons
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Convert class ID to integer
            if class_id == 0:  # Person class
                # Get bounding box coordinates for the detected person
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

                # Crop the detected face region
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:  # Skip if no face is detected
                    continue

                # Preprocess the face image for gender detection
                face_crop_preprocessed = preprocess_face(face_crop)

                # Apply gender detection
                gender_prediction = gender_model.predict(face_crop_preprocessed)[0]
                gender_idx = np.argmax(gender_prediction)
                gender_label = gender_classes[gender_idx]

                # Increment gender count
                if gender_label == 'Male':
                    male_count += 1
                    color = (255, 0, 0)  # Blue for male
                else:
                    female_count += 1
                    color = (0, 0, 255)  # Red for female

                # Draw bounding box around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display the gender label on the bounding box
                label = f'{gender_label}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the current frame with bounding boxes and labels
    cv2.imshow('Live Person and Gender Detection', frame)

    # Print gender count every 2 seconds
    if time.time() - last_print_time >= 2:
        print(f"Males: {male_count}, Females: {female_count}")
        last_print_time = time.time()  # Reset the timer

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
