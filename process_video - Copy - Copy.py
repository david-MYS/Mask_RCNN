import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names
import json


capture = cv2.VideoCapture('carvideo1.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked.avi', codec, 60.0, size)

count = 0
out_dict = {}

while(capture.isOpened()):
    # cap.read() returns a bool (True/False). 
    # If frame is read correctly, it will be True. So you can check end of the video by checking this return value
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        # put in dict
        # frame_excel = video length / 1/60 : r['class_ids']
        # CSV
        #out_dict[count] = r['class_ids']
        out_dict[count] = r 
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()


with open('frames.json', 'w') as fp:
   json.dump(out_dict, fp, indent=2)







