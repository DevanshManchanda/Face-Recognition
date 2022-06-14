import cv2
import pyautogui as pg

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    X = round(pg.position()[0]/8)
    Y = round(pg.position()[1]/4)
    Z = round((X+Y)/2)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(frame)
    if len(detections) >0:
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(X,Y,Z),2)
    
    cv2.resize(frame,(1920,1080), interpolation = cv2.INTER_AREA)
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
