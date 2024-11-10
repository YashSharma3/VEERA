import cv2
import requests
from ultralytics import YOLO
import time

# Function to send a POST request when a knife is detected
def send_sos_post_request():
    url = "http://13.126.70.254:8000/dashboard/api/sos-call/"
    try:
        response = requests.post(url, data = {
        "message": "M samasya m hu",
        "name": "DK",
        "phone_number": "9898989898",
        "email": "divyanshukhandelwal098@gmail.com",
        "latitude": 23.87,
        "longitude": 34.56
    })
        if response.status_code == 200:
            print("SOS POST request sent successfully!")
        else:
            print(f"Failed to send POST request, status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending POST request: {e}")

# Function to detect objects from the camera feed
def detect_objects_from_camera():
    # Initialize YOLO model
    yolo_model = YOLO('/Users/yashsharma/PycharmProjects/Veera/best.pt')

    # Open the default camera (0)
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Error opening camera"

    knife_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the camera feed using YOLOv8
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:  # Adjust the threshold as necessary
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]) * 10, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    # If 'knife' is detected, send a POST request and trigger the 10-second delay
                    if classes[int(cls[pos])] == 'knife':
                        send_sos_post_request()
                        knife_detected = True

        # Display the frame with bounding boxes and labels
        cv2.imshow('Object Detection - Live Feed', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If a knife is detected, introduce a 10-second delay and rerun the code
        if knife_detected:
            print("Knife detected. Rerunning detection in 10 seconds...")
            time.sleep(10)
            knife_detected = False  # Reset the flag and continue detection

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start object detection from the camera
detect_objects_from_camera()
