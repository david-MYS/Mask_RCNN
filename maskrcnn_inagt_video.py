"""
maskrcnn_inagt_video.py

Authors: MengYun (David) Shi (ms2979@cornell.edu)
         Nikolas Martelaro (nmartelaro@gmail.com)

Usage: maskrcnn_inagt_video.py

Purpose: Processes the Is Now a Good Time road video with MaskRCNN
         and creates a JSON file with all objects and bounding boxes
         for each frame of the video.
"""

import collections
import cv2
import glob
import json
import numpy as np
from visualize_cv2 import model, display_instances, class_names

def process_clip(video):
    ## Calculate video duration
    v = cv2.VideoCapture(video)
    v.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    duration = v.get(cv2.CAP_PROP_POS_MSEC)

    frameCount = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    print('duration = {}'.format(duration))
    print('number of frames = {}'.format(frameCount))

    # the 1st frame is frame 0, not 1, so "5335" means after the last frame
    POS_FRAMES = v.get(cv2.CAP_PROP_POS_FRAMES)
    FRAME_COUNT = v.get(cv2.CAP_PROP_FRAME_COUNT)
    print('POS_FRAMES = ' + str(POS_FRAMES))
    print('FRAME_COUNT = ' + str(FRAME_COUNT))

    v.release()

    ## Output video frame time, frame content, and bounding boxes to dictionary
    capture = cv2.VideoCapture(video)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30
    output = cv2.VideoWriter('{}_masked.avi'.format(video), codec, fps, (640, 360))

    count = 1
    out_dict = {}

    while(capture.isOpened()):
        # cap.read() returns a bool (True/False). 
        # If frame is read correctly, it will be True. So you can check end of the video by checking this return value
        ret, frame = capture.read()
        if ret:
            road_frame = frame[0:360, 640:1280]
            # add mask to frame
            results = model.detect([road_frame], verbose=0)
            r = results[0]
            road_frame = display_instances(
                road_frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
            )
            output.write(road_frame)
            
            out_dict[count] = {}
            out_dict[count]['cls_id'] = r['class_ids']
            out_dict[count]['rois'] = r['rois']

            print('[INFO] Frame {}/{}'.format(count,frameCount))
            count += 1
        else:
            break

    capture.release()
    output.release()

    ## Put the video data into a JSON file
    frame2content = {}
    for k, v in out_dict.items():
        objs = [class_names[i] for i in v["cls_id"]]
        obj2num = dict(collections.Counter(objs))
        _objs = []
        for o in objs:
            _objs.append(o + "_" + str(obj2num[o]))
            obj2num[o] -= 1
            
        obj2roi = {}
        for idx, o in enumerate(_objs):
            obj2roi[o] = list(map(int, list(v["rois"][idx, :])))
        frame2content[k] = obj2roi

    json_log = open('{}.json'.format(video), 'w')
    json.dump(frame2content, json_log)
    json_log.close()

video_dir = '/mnt/DATA/IsNowAGoodTime/INAGT_clean/m026/CLIPS/6_0_annotation_quad/'
for video in glob.glob('{}*_trimmed_audio.mov'.format(video_dir)):
    process_clip(video)


