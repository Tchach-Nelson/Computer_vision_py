import cv2
import time
import poseModule as mp



cap = cv2.VideoCapture('poseEstimation/poseVideos/video5.mp4')
pTime = 0
cTime = 0
detector = mp.PoseDetector()

while True:
    _, img = cap.read()
    img = cv2.resize(img, (700, 500))

    img = detector.findPose(img, draw=True)
    listLms = detector.findPosition(img, draw=True) 

    if len(listLms):  
        cv2.circle(img,( listLms[20][1], listLms[20][2] ), 10, (255,255,255), cv2.FILLED)
        cv2.circle(img,( listLms[19][1], listLms[19][2] ), 10, (255,255,255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

