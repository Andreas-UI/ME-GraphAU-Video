import cv2
import ffmpegcv
from download_checkpoints import download_checkpoints
from MEGraphAU.OpenGraphAU.predict import predict
from MEGraphAU.OpenGraphAU.utils import Image, draw_text
import json

def test_predict():
    try:
        download_checkpoints()
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

                # Early Stopping for test
                if current_time > 10:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
                faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
            
                for (x, y, w, h) in faces: 
                    faces = frame[y:y + h, x:x + w]
                    
                    infostr_aus, pred = predict(Image.fromarray(faces))

                    res, f = draw_text(frame, list(infostr_aus), pred, ( (x, y), (x+w, y+h)))
                    results[current_time] = res

                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
            
                output_frames.append(frame)
            
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

    except Exception as e:
        assert False, f"Exception occured: {e}"
