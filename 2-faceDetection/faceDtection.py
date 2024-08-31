import cv2
import mediapipe as mp
import time

#vous pouvez utiliser une source distance  ou une video local 
# cap = cv2.VideoCapture('2-faceDetection/videos/video7.mp4') 
cap = cv2.VideoCapture('http://192.168.137.254:8080/video')

pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection( min_detection_confidence=0.75, model_selection=1)



while True:

    _, img = cap.read()
    img = cv2.resize(img, (800, 700))
    # img = cv2.resize(img, (700, 750))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(imgRGB)
    # print(result)
    if result.detections:
        for id, detection in enumerate(result.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape 
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255,255,255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow('Image', img)

    # Réduire la fréquence d'affichage
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
