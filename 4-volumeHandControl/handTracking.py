import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture(1)  # code pour webcam Webcam
#recuperation de la video camera du tel
cap = cv2.VideoCapture('http://192.168.1.37:8080/video')

# declaration 
mpHands = mp.solutions.hands 
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
mpDraw = mp.solutions.drawing_utils 

#variable de compte des fps (frame par seconde)
pTime = 0
cTime = 0

while(True):

    #recuperation de la frame (images) sur video
    _, img = cap.read()

    #dimunier la taille pour plus de vitesse
    img = cv2.resize(img, (500, 400)) 

    #obtention des marqueurs de main "hands.process().multi_hand_landmarks" a partir des frames(images) converties en RGB "cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).multi_hand_landmarks 

    if results:
        for handLms in results:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # if id == 4:
                cv2.circle(img, (cx, cy), 10, (255,255,255), cv2.FILLED)
                    
            #ajout ou dessinage des marqueurs "handLms" sur l'image "img" :: "mpDraw.draw_landmarks(img, handLms)" et ajout des liens entre marqueurs avec "mpHands.HAND_CONNECTIONS" 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    #calcul des frames (images) par seconde (fps)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #ajout de fps converti en entier int(fps) puis en chaine str(int(fps)) sur l'image (img) avec pour position (10,70) et couleur (255,255,255) 
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
    
    #affichage des images (+ frame et +marqueur) sur une fenetre win
    cv2.imshow('Camera',img)

    #touche q pour quitter
    if cv2.waitKey(1) == ord("q"):
        break

#stopper les cameras et fermer la fenetre
cap.release()
cv2.destroyAllWindows()