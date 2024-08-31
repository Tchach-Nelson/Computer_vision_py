import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

# cap = cv2.VideoCapture(1)  # code pour webcam Webcam
#recuperation de la video camera du tel
cap = cv2.VideoCapture('http://192.168.74.92:8080/video')
    
pTime = 0
cTime = 0
detector = htm.handDetector()

while(True):

    _, img = cap.read()

    img = cv2.resize(img, (500, 400)) 

    img = detector.findHand(img, draw=True)   
    lmsList = detector.findPosition(img, draw=True)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
    
    cv2.imshow('Camera',img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()