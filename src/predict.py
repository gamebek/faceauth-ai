import cv2
import pickle
import numpy as np

# Load mode
with open("models/face_model.pkl", "rb") as f:
    model = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess(face):
    face = cv2.resize(face, (64, 64))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    return face.flatten().reshape(1, -1)

cap = cv2.VideoCapture(0)

threshold = 0.6 

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed = preprocess(face)

        distances, _ = model.kneighbors(processed)
        distance = distances[0][0]

        if distance > threshold:
            label = "Access Denied"
            color = (0, 0, 255)
        else:
            label = model.predict(processed)[0]
            color = (0, 255, 0)

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Login", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()