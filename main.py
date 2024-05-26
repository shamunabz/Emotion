# main.py
import cv2
from deepface import DeepFace
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame with DeepFace
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Get the largest face detected (DeepFace might detect multiple faces)
        if isinstance(results, list):
            result = max(results, key=lambda r: r['region']['w'] * r['region']['h'])
        else:
            result = results
        
        # Get bounding box and emotion
        region = result['region']
        emotion = result['dominant_emotion']

        # Draw a box around the face
        top, right, bottom, left = region['y'], region['x'] + region['w'], region['y'] + region['h'], region['x']
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the emotion below the face
        label = f"Emotion: {emotion}"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
