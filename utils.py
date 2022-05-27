import datetime 
import cv2
import csv

def face_extractor(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        # frame = cv2.putText(img, name, (x, y-4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

def update_attendance(name):
    
    with open('attendance.csv', "a", newline="") as fObject:
        writer = csv.writer(fObject)
        cur_time = str(datetime.datetime.now())
        writer.writerow([name, str(cur_time)])
