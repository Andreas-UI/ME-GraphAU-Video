import cv2
import ffmpegcv
from MEGraphAU.OpenGraphAU.predict import predict
from MEGraphAU.OpenGraphAU.utils import Image, draw_text
import json
from ultralytics import YOLO


yolo = YOLO("yolov8n-face.pt")

video_path = "videos/v_ArmFlapping_01.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

output_frames = []
results = {}
 
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = frame_number / fps

        faces = yolo.predict(frame, conf=0.40, iou=0.3)
        for face in faces:
            parameters = face.boxes

            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2 - y1, x2 - x1
                faces = frame[y1:y1 + h, x1:x1 + w] 

                infostr_aus, pred = predict(Image.fromarray(faces))
                res, f = draw_text(frame, list(infostr_aus), pred, ( (x1, y1), (x1+w, y1+h)))
                results[current_time] = res

                frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2) 

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4) 

        # for (x, y, w, h) in faces: 
        #     faces = frame[y:y + h, x:x + w] 
        #     # Display the resulting frame
        #     # cv2.imshow('Frame',frame)
        #     infostr_aus, pred = predict(Image.fromarray(faces))

        #     res, f = draw_text(frame, list(infostr_aus), pred, ( (x, y), (x+w, y+h)))
        #     # cv2.imshow("frame", f)

        #     results[current_time] = res

        #     frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 

        output_frames.append(frame)
        cv2.imshow("frame", frame)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Break the loop
    else: 
        break

cap.release()

size = output_frames[0].shape
output_video = ffmpegcv.VideoWriter(f"{video_path[:-4]}_output.mp4", None, fps)

for of in output_frames:
    output_video.write(of)
output_video.release()

with open(f"{video_path[:-4]}_output.json", 'w') as f:
    json.dump(results, f)
