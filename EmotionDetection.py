import cv2
import numpy as np
from keras.models import model_from_json

flag = 0
happy_count = 0
sad_count = 0
neutral_count = 0
face_front = 0
face_flag = 0
frame_count = 0

emotion_dict = {0: "Happy", 1: "Neutral", 2: "Sad"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')   
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.3, minNeighbors=5)
    # print (face_rects)
    if face_rects == (): return 0
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,255,0),5)

    return (x,y,w,h)

while(True):

    if flag == 1: break

    tracker = cv2.legacy.TrackerKCF_create()

    tracker_name = str(tracker).split()[0][8:]

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1080, 720))
    print("Searching face...")
    while detect_face(frame) == 0:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1080, 720))
        print("Searching face...")
        if face_front > 0 and face_flag == 1: 
            face_front -= 1

    roi = detect_face(frame)
    print("Face Detected!")
    face_flag = 1
    print(roi)
    (x,y,w,h) = tuple(map(int,roi))
    if x < 100 or y < 100 or w < 100 or h < 100: continue
    roi1 = x+50, y+50, w-100, h-100
    ret = tracker.init(frame, roi1)


    while True:
        try:
            frame_count += 1

            ret, frame = cap.read()
            frame = cv2.resize(frame, (1080, 720))
            # frame = cv2.flip(frame, 1)

            success, roi = tracker.update(frame)

            (x,y,w,h) = tuple(map(int,roi))
            
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.rectangle(frame, (x, y-50), (x+w, y+h), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            if(len(roi_gray_frame)>0):
                cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), 0)

            if len(cropped_img) > 0:
                face_front += 1

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if emotion_dict[maxindex] == 'Happy':
                happy_count += 1
            elif emotion_dict[maxindex] == 'Sad':
                sad_count += 1
            elif emotion_dict[maxindex] == 'Neutral':
                neutral_count += 1

                
            # Draw Rectangle as Tracker moves
            if success:
                # Tracking success
                p1 = (x, y)
                p2 = (x+w, y+h)
                cv2.rectangle(frame, p1, p2, (0,0,255), 3)
            else :
                # Tracking failure
                cv2.putText(frame, "Failure to Detect Tracking!!", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
                face_front -= 1
                break
                

            # Display tracker type on frame
            cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)


            avg_emotion = (happy_count * 0.9 + sad_count * 0.8 + neutral_count) / frame_count

            front_view = face_front/frame_count

            attention = (avg_emotion * front_view) * 100

            cv2.putText(frame, "Average Emotion Index = ", (0,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3) 

            cv2.putText(frame, str(round(avg_emotion,3)), (500,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)  

            cv2.putText(frame, "Attention Rate = " , (0,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3) 

            cv2.putText(frame, str(round(attention, 3)), (500,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)  

            if (attention > 80.0): 
                cv2.putText(frame, "Attentive" , (800,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

            elif (attention > 65.0) and (attention < 80.0) :
                cv2.putText(frame, "Nearly Attentive" , (800,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

            elif (attention > 45.0) and (attention < 65.0): 
                cv2.putText(frame, "Bearly Attentive" , (800,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

            else: 
                cv2.putText(frame, "Inattentive" , (800,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)

            cv2.imshow("Attention Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                flag = 1
                break

        except Exception as e:
            print(e)

    
cap.release()
cv2.destroyAllWindows()

print("Total Frames", frame_count)

print("Total Front View", face_front)

print("Happy = ", round(happy_count/frame_count*100, 3))

print("Sad = ", round(sad_count/frame_count*100, 3))

print("Neutral = ", round(neutral_count/frame_count*100, 3))

avg_emotion = (happy_count * 0.9 + sad_count * 0.8 + neutral_count) / frame_count

print("Average Emotion Index = ", round(avg_emotion,3))

front_view = face_front/frame_count

print("Front View Rate = ", round(front_view, 3))

attention = (avg_emotion * front_view) * 100

print("Attention Rate = ", round(attention, 3))


