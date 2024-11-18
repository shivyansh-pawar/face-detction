import cv2

# Initialize video capture and load the Haar cascade classifier
cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade classifier is loaded correctly
if cascade_classifier.empty():
    print("Error: Cascade Classifier not loaded.")
    exit()

while True:
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale for face detection (this is required for Haar cascades)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    detections = cascade_classifier.detectMultiScale(gray_frame, 1.3, 5)

    # If faces are detected, draw rectangles around them
    if len(detections) > 0:
        for (x, y, w, h) in detections:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows when done
cap.release()
cv2.destroyAllWindows()
