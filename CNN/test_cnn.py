import numpy as np
import cv2
import time
import torch
from my_modules import VanillaCNN
from tools import get_features_from_coordinates
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--source",help="num Bus for camera and 0 for video file",type=int,required=True)
parser.add_argument("--file",help="filename for video")
parser.add_argument("--classes",help="number of classes",type=int,required=True)
parser.add_argument("--thresh",help="threshold",type=int,required=True)
args = parser.parse_args()
cap = None
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

videofilename = ""
if(args.source < 0):
    videofilename = args.file
    if(os.path.isfile(videofilename)):
        cap = cv2.VideoCapture(videofilename)
    else:
        print("Video file does not exist")
        sys.exit()
elif(args.source >= 0):
    cap = cv2.VideoCapture(args.source)
else:
    print("Exiting")
    sys.exit()

model = VanillaCNN((1,80,80),args.classes,10**-5)
model = model.to(device) 
mean_tensor = torch.load("data/mean_tensor.pt")
std_tensor = torch.load("data/std_tensor.pt")
model.load_state_dict(torch.load('data/model_cnn2.pt'))
model.eval()


#cap = cv2.VideoCapture("data/full_test.avi")
#cap = cv2.VideoCapture(4)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('CNN_results.avi', fourcc, 15.0, (640,480))
thresh = args.thresh
#load classifier and PCA + Load Classifier
while(cap.isOpened()):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        rows,cols,_ = frame.shape
    except:
        break
    mask = np.zeros((rows,cols))
    B = frame[:,:,0]
    G = frame[:,:,1]
    R = frame[:,:,2]
    cond1 = np.logical_and(R>=G,R>=B)
    cond2 = np.logical_or(R - G > thresh, R - B > thresh)
    mask[np.logical_and(cond1,cond2)] = 255
    mask = np.uint8(mask)
    cv2.medianBlur(mask,7,dst=mask)
    cv2.medianBlur(mask,7,dst=mask)
    edges = cv2.Canny(mask,100,300)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        max_len = len(contours[0])
    except:
        pass
    max_i = 0
    i = 0
    try:
        for c in contours :
            if(max_len < len(c)):
                max_len = len(c)
                max_i = i
            i = i + 1
    except:
        pass
    try:
        hand_contour = contours[max_i].reshape(-1,2)
        max_x = np.amax(hand_contour[:,0])
        max_y = np.amax(hand_contour[:,1])
        min_x = np.amin(hand_contour[:,0])
        min_y = np.amin(hand_contour[:,1])
        cv2.rectangle(frame, (min_x,min_y),(max_x,max_y), (255,255,0), thickness=1, lineType=8, shift=0)
        imroi = frame[min_y:max_y, min_x:max_x]
        imroi = cv2.resize(imroi,(80,80))
        imroi = cv2.cvtColor(imroi,cv2.COLOR_BGR2GRAY)
        normalizedImg = np.zeros((80,80))
        imroi = cv2.normalize(imroi,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        #cv2.drawContours(frame,contours,max_i,(255,0,0))
        imroi2 = torch.from_numpy(imroi).float().to(device)/255
        imroi2 = imroi2.unsqueeze(0)
        imroi2 = (imroi2 - mean_tensor)/std_tensor
        imroi2 = imroi2.unsqueeze(0)
        #print(imroi2.max())
        ppred = model(imroi2)
        #print(pred)#, pred.size())
        pred = int(ppred.argmax(dim=1)[0]) 
        print(ppred, pred)
        text = str(pred)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,text,(100,100), font, .8,(255,255,255),2,cv2.LINE_AA)
        #print(pred[0])
        
        cv2.imshow('ROI',imroi)
        out.write(frame)
        #cv2.imshow('mask',mask)
        #cv2.imshow('edges',edges)
        #time.sleep(0.08)
    except:
        pass
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()