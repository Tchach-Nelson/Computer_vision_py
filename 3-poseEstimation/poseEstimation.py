import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture('3-poseEstimation/poseVideos/video3.mp4')
cap = cv2.VideoCapture('http://192.168.137.254:8080/video')

pTime = 0
cTime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=0,  #Réduire la complexité du modèle
                   smooth_landmarks=True,
                   enable_segmentation=False, 
                   smooth_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()

    img = cv2.resize(img, (700, 500))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB).pose_landmarks

    if result:
        mpDraw.draw_landmarks(img, result, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # cv2.circle(img,(cx,cy), 1, (255,0,0), cv2.FILLED)

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