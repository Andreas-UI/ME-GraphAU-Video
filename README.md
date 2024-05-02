# About
Our project applies a specific algorithm from a [research paper](https://arxiv.org/abs/2205.01782) to predict Facial Action Units (FAUs). Although the original paper focuses on still images, we've adapted the algorithm to predict FAUs in videos, analyzing each frame individually for face detection.

For our implementation, we've chosen to use the stage 2 of the ResNet50 architecture. It's important to note that modifying parameters may lead to errors due to the specialized nature of our implementation.

# Demonstration
## Original Video
This video is sourced from the Self-Stimulatory Behavior Dataset (SSBD) [[link](https://ieeexplore.ieee.org/document/6755972)], focusing on autism behavior. Specifically, it depicts a boy doing arm-flapping behavior.


https://github.com/Andreas-UI/ME-GraphAU-Video/assets/51452532/0034e415-edcb-47ff-abe5-6efe91c70093


<video width="320" height="240" controls>
  <source src="videos/v_ArmFlapping_01.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Output Video
The output video contains the facial action units (FAU) generated by the paper along with a bounding box with emotion on the face.


https://github.com/Andreas-UI/ME-GraphAU-Video/assets/51452532/254fc417-c8b9-468c-8d18-20335474ae33


<video width="320" height="240" controls>
  <source src="videos/demo_v_ArmFlapping_01_output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# How to Use
## Checkpoints
Download and insert both of these checkpoints to MEGraphAU > OpenGraphAU > checkpoints.
* resnet50-19c8e357.pth : [link](https://download.pytorch.org/models/resnet50-19c8e357.pth)
* OpenGprahAU-ResNet50_second_stage.pth : [link](https://drive.google.com/file/d/1UMnpbj_YKlqHF1m0DHV0KYD3qmcOmeXp/view?usp=sharing)

Your folder structure should look something like this.<br>
![alt text](image.png)

## Code
```python
# Import Libraries
import cv2
import ffmpegcv
from MEGraphAU.OpenGraphAU.predict import predict
from MEGraphAU.OpenGraphAU.utils import Image, draw_text
import json

video_path = "videos/v_ArmFlapping_01.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

output_frames = []
results = {}

# Read video frame by frame.
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = frame_number / fps

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
    
        for (x, y, w, h) in faces: 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
            faces = frame[y:y + h, x:x + w]

            # Predict the face on this frame.
            infostr_aus, pred = predict(Image.fromarray(faces))

            # Draw the results
            res, f = draw_text(frame, list(infostr_aus), pred, ( (x, y), (x+w, y+h)))
            cv2.imshow("frame", f)

            results[current_time] = res
            output_frames.append(f)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else: 
        break

cap.release()

# OPTIONAL: Save output video
size = output_frames[0].shape
output_video = ffmpegcv.VideoWriter(f"{video_path[:-4]}_output.mp4", None, fps)

for of in output_frames:
    output_video.write(of)
output_video.release()

# OPTIONAL: Save FAU results 
with open(f"{video_path[:-4]}_output.json", 'w') as f:
    json.dump(results, f)

```

🎓 Citation
=
if the code or method help you in the research, please cite the following paper:
```

@inproceedings{luo2022learning,
  title     = {Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition},
  author    = {Luo, Cheng and Song, Siyang and Xie, Weicheng and Shen, Linlin and Gunes, Hatice},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  pages     = {1239--1246},
  year      = {2022}
}


@article{song2022gratis,
    title={Gratis: Deep learning graph representation with task-specific topology and multi-dimensional edge features},
    author={Song, Siyang and Song, Yuxin and Luo, Cheng and Song, Zhiyuan and Kuzucu, Selim and Jia, Xi and Guo, Zhijiang and Xie, Weicheng and Shen, Linlin and Gunes, Hatice},
    journal={arXiv preprint arXiv:2211.12482},
    year={2022}
}



```
