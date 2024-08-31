import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon, model_complexity=0)
        self.mpDraw = mp.solutions.drawing_utils 


    def findHand(self, img, draw=True):

        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).multi_hand_landmarks 

        if self.results:
            for handLms in self.results:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
    
        return img
    
    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results:

            myHand = self.results[handNo]

            for id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # if id == 4:
                    cv2.circle(img, (cx, cy), 7, (255,255,255), cv2.FILLED)

        return lmList
                    
def main():

    # cap = cv2.VideoCapture(1)  # code pour webcam Webcam
    #recuperation de la video camera du tel
    cap = cv2.VideoCapture('http://192.168.74.92:8080/video')
        
    pTime = 0
    cTime = 0
    detector = handDetector()

    while(True):

        _, img = cap.read()

        img = cv2.resize(img, (500, 400)) 

        img = detector.findHand(img)
        lmsList = detector.findPosition(img)
        # if len(lmsList != 0):
        #     print(lmsList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        
        cv2.imshow('Camera',img)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()