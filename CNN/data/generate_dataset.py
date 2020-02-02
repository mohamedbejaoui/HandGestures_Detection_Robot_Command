import numpy as np
import cv2
import sys
import os

labels = []
files = []
Dat = []

for i,arg in enumerate(sys.argv):
    if(i>0):
        if(i%2 == 0):
            #we have a label
            labels.append(int(arg))
        else:
            #we have a file
            if(not os.path.isfile(arg)):
                print("{} does not exist".format(arg))
            else:
                files.append(arg)
if(len(labels) != len(files)):
    print("Data not coherente")
    sys.exit()

for _i,f in enumerate(files):
    cap = cv2.VideoCapture(f)
    thresh = 15
    k = 0
    while(cap.isOpened()):
        k = k + 1
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
        cond2 = np.logical_or(R - G > 25, R - B > 25)
        mask[np.logical_and(cond1,cond2)] = 255
        mask = np.uint8(mask)
        cv2.medianBlur(mask,7,dst=mask)
        cv2.medianBlur(mask,7,dst=mask)
        edges = cv2.Canny(mask,100,300)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        max_len = len(contours[0])
        max_i = 0
        i = 0
        for c in contours :
            if(max_len < len(c)):
                max_len = len(c)
                max_i = i
            i = i + 1
        
        hand_contour = contours[max_i].reshape(-1,2)
        max_x = np.amax(hand_contour[:,0])
        max_y = np.amax(hand_contour[:,1])
        min_x = np.amin(hand_contour[:,0])
        min_y = np.amin(hand_contour[:,1])
        cv2.rectangle(frame, (min_x,min_y),(max_x,max_y), (255,255,0), thickness=1, lineType=8, shift=0)
        imroi = frame[min_y:max_y, min_x:max_x]
        imroi = cv2.resize(imroi,(80,80))
        imroi = cv2.cvtColor(imroi,cv2.COLOR_BGR2GRAY)
        normalizedImg = np.zeros((80, 80))
        imroi = cv2.normalize(imroi,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow('frame',frame)
        cv2.imshow('ROI',imroi)
        #cv2.imshow('mask',mask)
        #cv2.imshow('edges',edges)
        imgname = "images/{}_{}.jpg".format(os.path.splitext(f)[0],k)
        Dat.append([imgname,labels[_i]])
        cv2.imwrite(imgname,imroi)
        #time.sleep(0.08)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
Dat = np.asarray(Dat)
print(Dat)
#np.savetxt('meta_cnn.csv',Dat,delimiter=',')

np.savetxt('meta_cnn.csv', Dat, delimiter=',', header='', comments='', fmt='%s')
