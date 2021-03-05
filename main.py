import cv2

from cv2 import WINDOW_NORMAL
from face_detection import faces_find

#initializing escape key
ESC = 27

#initializing webcam
def begin_webcame(emotion_model, gender_model, win_size, win_name='Realtime', updatetime=50):
    cv2.namedWindow(win_name, WINDOW_NORMAL)
    if win_size:
        width, height = win_size
        cv2.resizeWindow(win_name, width, height)

    realtime_feed = cv2.VideoCapture(0)
    realtime_feed.set(3, width)
    realtime_feed.set(4, height)
    read_val, frame = realtime_feed.read()

    wait_time = 0 #delay bewteen the frames is zero
    init = True
    while read_val:
        read_val, frame = realtime_feed.read()
        for normal_face, (x, y, w, h) in faces_find(frame):  #finding the faces in faces_find function initiated in face_detection.py
          if init or wait_time == 0:
            init = False
            prediction_emotion = emotion_model.predict(normal_face)
            prediction_gender = gender_model.predict(normal_face)
          if (prediction_gender[0] == 0):
              cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2) #blue color for female
          else:
              cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #red color for male
          cv2.putText(frame, emotions[prediction_emotion[0]], (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (255, 0, 0), 2)
          cv2.putText(frame, gender[prediction_gender[0]], (10,20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,
                      (255, 0, 0), 2)

        wait_time += 1
        wait_time %= 20
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(updatetime)
        if key == ESC:
            break

    cv2.destroyWindow(win_name)



if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    gender   =  ["male","female"]
    # Loading the trained models of emotion
    fisher_face_emo = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emo.read('E:\models\emotion_classifier_model.xml')

#loading the trained models of gender
    fisher_face_gen = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gen.read('E:\models\gender_classifier_model.xml')

    # starting the model to predict
    choice = input("starting your webcam?(y/n) ")
    if (choice == 'y'):
        window_name = "Facifier Webcam (press ESC to exit)"
        begin_webcame(fisher_face_emo, fisher_face_gen, win_size=(1280, 720), win_name=window_name, updatetime=15)

    else:
        print("Invalid input, exiting program.")

