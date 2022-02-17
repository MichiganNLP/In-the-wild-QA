import argparse
import os
import time

import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from torchvision import transforms

from models import *
from sort import *


def parse_args():
    parser = argparse.ArgumentParser()

    # Common arguments among models
    parser.add_argument('--video_path',
        help='path to video')

    parser.add_argument('--out_vid_dir',
        help='output directory for the video file')
    
    parser.add_argument('--output_dir',
        help='output directory for the detected object file')

    args = parser.parse_args()
    return args

args = parse_args()

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = args.video_path
clip = VideoFileClip(videopath)
vid_duration = clip.duration


colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

# cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
out_vid_dir = args.out_vid_dir
out_vid_path = os.path.join(out_vid_dir, os.path.basename(videopath.replace(".mp4", "-det.avi")))
outvideo = cv2.VideoWriter(out_vid_path,fourcc,20.0,(vw,vh))

detect_objs = defaultdict(dict)

frames = 0
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]    # object name
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            obj_name = cls + "-" + str(int(obj_id))
            if obj_name not in detect_objs:
                detect_objs[obj_name]['start'] = frames
                detect_objs[obj_name]['end'] = frames
            else:
                detect_objs[obj_name]['end'] = frames   # as long as the object appears before, update the end frame

    # cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

# convert detection frames to time
f_per_t = vid_duration / frames
for obj_name, _ in detect_objs.items():
    detect_objs[obj_name]['start'] *= f_per_t
    detect_objs[obj_name]['end'] *= f_per_t

# write the detected information to a file
with open(f'{args.output_dir}/{os.path.basename(videopath).strip(".mp4")}.txt', 'w') as f:
    for obj_name, _ in detect_objs.items():
        st, ed = detect_objs[obj_name]['start'], detect_objs[obj_name]['end']
        f.write(f"{obj_name} : {st} - {ed}\n")

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
# cv2.destroyAllWindows()
outvideo.release()
