import cv2
import numpy as np
from keras.models import load_model

classifier=cv2.CascadeClassifier(r'\haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0)

#load the learned data (convo_base)
model=load_model(r'conv_base.h5')
exp=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

while True:
    res,frame=video.read()
    if res:
        #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=classifier.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5)
        for (a,b,c,d) in faces:
            req_face=frame[b:b+d,a:a+c]
            req_face=cv2.resize(req_face,(48,48))
            req_face=np.expand_dims(req_face,axis=0)

            y=model.predict(req_face)
            y=np.argmax(y)
            print(exp[y])
            cv2.rectangle(frame,(a,b),(a+c,b+d),(255,0,0),2)
            cv2.putText(frame,f'given expression is {exp[y]}',(a-50,b),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),2)
            cv2.imshow('frames',frame)

    else:
        break

    if cv2.waitKey(1)&0xFF==ord('q'):
       break




