import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###########################################
wCam, hCam = 500, 400 
###########################################

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('http://192.168.137.254:8080/video')
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7) 


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel() 
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol=0 
volBar=350
volPer=0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (500, 400))  
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        #recuperation de la position de index et du pouce 
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        #calcul du centre
        cx, cy = (x1 + x2) //2, (y1 + y2) //2 

        #affichage sur l'img
        cv2.circle(img, (x1, y1), 10, (255,255,255), cv2.FILLED)   
        cv2.circle(img, (x2, y2), 10, (255,255,255), cv2.FILLED)   
        cv2.line(img, (x1, y1), (x2,y2), (255,255,255),3)
        cv2.circle(img, (cx, cy), 8, (255,255,255), cv2.FILLED)
     
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        #hand range 50 - 300
        #volume Range -95.25  - 0
        vol = np.interp(length,[50,300], [minVol, maxVol])
        volBar = np.interp(length,[50,300], [350, 150])
        volPer = np.interp(length,[50,300], [0, 100])

        # print(vol) 
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 8, (0,255,0), cv2.FILLED)


    cv2.rectangle(img,(20,100), (60, 350), (255,255,255), 3 )
    cv2.rectangle(img,(20,int(volBar)), (60, 350), (255,255,255), cv2.FILLED )
    cv2.putText(img, f'{int(volPer)}%', (15,380), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
    cv2.imshow("Img", img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()