import cv2
import mediapipe as mp
import time

class PoseDetector():

    def __init__(self, mode=False, complexity= 0, smoothLms=True, enableSeg=False, smoothSeg=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothLms = smoothLms
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,  #Réduire la complexité du modèle
            smooth_landmarks=self.smoothLms,
            enable_segmentation=self.enableSeg, 
            smooth_segmentation=self.smoothSeg,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB).pose_landmarks

        if self.result:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result, self.mpPose.POSE_CONNECTIONS)

        return img
        
    def findPosition(self, img, draw=True):

        lmList = [] 

        if self.result:
            for id, lm in enumerate(self.result.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img,(cx,cy), 1, (255,0,0), cv2.FILLED)

        return lmList



def main():
    cap = cv2.VideoCapture('poseEstimation/poseVideos/video5.mp4')
    pTime = 0
    cTime = 0
    detector = PoseDetector()

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

if __name__ == "__main__":
    main()