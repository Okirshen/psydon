import cv2
import time
import os

# variables
timer = 5
people = 1


last_picture = timer


def path(string: str) -> str:
    return os.path.join(os.path.dirname(__file__), string)


face_cascade = cv2.CascadeClassifier(path('face.xml'))

video_capture = cv2.VideoCapture(0)
face_locations = []

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) >= people and time.time() - last_picture >= timer:
        print('taking picture')
        blur = cv2.Laplacian(frame, cv2.CV_64F).var()
        print('blur amount:', blur)
        cv2.imwrite(
            path(f'img/{time.strftime("%d-%m-%Y_%H:%M:%S")}_{round(blur)}.png'), frame)
        last_picture = time.time()

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
