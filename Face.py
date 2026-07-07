import cv2
import face_recognition

video_capture = cv2.VideoCapture(2)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert image from BGR color (OpenCV format) to RGB color (Face Recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow('Face Detection Sandbox', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()