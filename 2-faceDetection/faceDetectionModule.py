import cv2
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self, minDetection=0.75, mode=1):

        self.minDetection = minDetection
        self.mode = mode

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection( min_detection_confidence=minDetection, model_selection=mode)

    def findFace(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        # print(self.result)
        bboxs = []

        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape 
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                
                bboxs = [id, bbox, detection.score]

                if draw:
                    img = self.fancyDraw(img, bbox)

                    cv2.rectangle(img, bbox, (255,255,255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

   
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):

        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # top left
        cv2.rectangle(img, bbox, (255,255,255), rt)
        cv2.line(img, (x,y), (x + l, y), (255, 255, 255), t)
        cv2.line(img, (x,y), (x, y + l), (255, 255, 255), t)
        # top right
        cv2.rectangle(img, bbox, (255,255,255), rt)
        cv2.line(img, (x1,y), (x1 - l, y), (255, 255, 255), t)
        cv2.line(img, (x1,y), (x1, y + l), (255, 255, 255), t)
        # bottom left
        cv2.rectangle(img, bbox, (255,255,255), rt)
        cv2.line(img, (x,y1), (x + l, y1), (255, 255, 255), t)
        cv2.line(img, (x,y1), (x, y1 - l), (255, 255, 255), t)
        # bootom left 
        cv2.rectangle(img, bbox, (255,255,255), rt)
        cv2.line(img, (x1,y1), (x1 - l, y1), (255, 255, 255), t)
        cv2.line(img, (x1,y1), (x1, y1 - l), (255, 255, 255), t)

        return img

def main():

    cap = cv2.VideoCapture('faceDetection/videos/video1.mp4')
    pTime = 0
    cTime = 0
    detector = FaceDetector()

    while True:

        _, img = cap.read()
        img = cv2.resize(img, (800, 700))

        img, bboxs =detector.findFace(img)

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

if __name__ == "__main__":
    main()