import cv2
import time

# variables
timer = 5
people = 1


last_picture = timer

face_cascade = cv2.CascadeClassifier('face.xml')

video_capture = cv2.VideoCapture(0)
face_locations = []

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) >= people and time.time() - last_picture >= timer:
        print('taking picture')
        print('blur amount:', cv2.Laplacian(frame, cv2.CV_64F).var())
        cv2.imwrite(
            f'img/{time.strftime("%d-%m-%Y_%H:%M:%S")}_{round(cv2.Laplacian(frame, cv2.CV_64F).var())}.png', frame)
        last_picture = time.time()

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
