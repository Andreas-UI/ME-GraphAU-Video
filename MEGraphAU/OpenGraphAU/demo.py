import os
import numpy as np
import torch
import torch.nn as nn
import logging
from dataset import pil_loader
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
import cv2
import json
import ffmpegcv
from predict import predict


def main(conf):
    dataset_info = hybrid_prediction_infolist

    # data
    video_path = conf.input
    results = {}

    # def predict(img):
    #     if conf.stage == 1:
    #         from model.ANFL import MEFARG
    #         net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    #     else:
    #         from model.MEFL import MEFARG
    #         net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)
        
    #     # resume
    #     if conf.resume != '':
    #         logging.info("Resume form | {} ]".format(conf.resume))
    #         net = load_state_dict(net, conf.resume)


    #     net.eval()
    #     img_transform = image_eval()
    #     # img = pil_loader(img_path)
    #     # img_ = img_transform(torch.from_numpy(img)).unsqueeze(0)
    #     img_ = img_transform(img).unsqueeze(0)

    #     if torch.cuda.is_available():
    #         net = net.cuda()
    #         img_ = img_.cuda()

    #     with torch.no_grad():
    #         pred = net(img_)
    #         pred = pred.squeeze().cpu().numpy()


    #     # log
    #     # infostr = {'AU prediction:'}
    #     # logging.info(infostr)
    #     infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
    #     # logging.info(infostr_aus)
    #     # logging.info(infostr_probs)

    #     return infostr_aus, pred
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

 
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    output_frames = []
 
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Calculate the current frame number
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Calculate the time of the current frame
            current_time = frame_number / fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 
            faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
        
            for (x, y, w, h) in faces: 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
                faces = frame[y:y + h, x:x + w] 
                # Display the resulting frame
                # cv2.imshow('Frame',frame)
                infostr_aus, pred = predict(Image.fromarray(faces), conf)

                res, f = draw_text(frame, list(infostr_aus), pred, ( (x, y), (x+w, y+h)))
                cv2.imshow("frame", f)

                results[current_time] = res

                # output_video.write(f)
                output_frames.append(f)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else: 
            break

    cap.release()

    size = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(size)
    # output_video = cv2.VideoWriter(
    #     "output.mp4", fourcc, 1, (size[0], size[1]))
    
    output_video = ffmpegcv.VideoWriter("output.mp4", None, 30)
    
    for of in output_frames:
        output_video.write(of)
    output_video.release()

    with open(video_path+'_output.json', 'w') as f:
        json.dump(results, f)


    # if conf.draw_text:
        # img = draw_text(conf.input, list(infostr_aus), pred)
    #     import cv2
    #     path = conf.input.split('.')[0]+'_pred.jpg'
    #     cv2.imwrite(path, img)


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)


# python demo.py --arc resnet50 --stage 2 --exp-name demo --resume checkpoints/OpenGprahAU-ResNet50_second_stage.pth --input demo_imgs/v_ArmFlapping_01.mp4
# python ME-GraphAU/OpenGraphAU/demo.py --arc resnet50 --stage 2 --exp-name demo --resume ME-GraphAU/OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_second_stage.pth --input v_ArmFlapping_01.mp4