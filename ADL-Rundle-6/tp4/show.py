import cv2
import time

FPS = 30

cap = cv2.VideoCapture("video.avi")
ret, frame = cap.read()
while ret:
    time.sleep(1 / FPS)
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
