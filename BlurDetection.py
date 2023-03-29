import cv2

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Loop over the video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Blur each detected face region
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (25, 25), 0)
        frame[y:y+h, x:x+w] = blurred_face

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite("try.jpg",frame)

    # Wait for a key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()