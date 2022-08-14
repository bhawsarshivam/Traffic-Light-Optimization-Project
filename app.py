from multiprocessing.connection import wait
from tkinter import *
from turtle import left
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

# Load Coco.names data set from yolo-coco repository
labelsPath = os.path.join('yolo-coco','coco.names')
LABELS = open(labelsPath).read().strip().split('\n')

# Load weights and configuration files form yolo-coco
weightPath = os.path.join('yolo-coco','yolov3.weights')
configPath = os.path.join('yolo-coco','yolov3.cfg')

# Load YOLO object detector model
net = cv2.dnn.readNetFromDarknet(configPath,weightPath)

# determine only the *output* layer name that we need from yolo
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


def getBoundingBox(imageF,result):
    boxes = []
    confidences = []
    classIDs = []
    
    (H,W) = imageF.shape[:2]
    
    for output in result:
        for detection in output:
            score = detection[6:13]
            classID = np.argmax(score) 
            confidence = score[classID]
            
            if confidence > 0.7:
                box = detection[:4] * np.array([W,H,W,H])
                (centerX,centerY,width,height) = box.astype('int')
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Now this particular line is use to keep high probablity boxes only
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)
    
    return idxs,boxes

def button1():
    dataNameLabel['text'] = 'Data\\vid1.mp4'
    videoSourceLabel['text'] = 'Video Source: YouTube'
    mainScreen(dataNameLabel['text'],num=1)

def button2():
    dataNameLabel['text'] = 'Data\\vid2.mp4'
    videoSourceLabel['text'] = 'Video Source: YouTube'
    mainScreen(dataNameLabel['text'],num=2)

def button3():
    dataNameLabel['text'] = 'Data\\vid3.mp4'
    videoSourceLabel['text'] = 'Video Source: YouTube: Mumbai Traffic'
    mainScreen(dataNameLabel['text'],num=3)


def mainScreen(videolink='',num=0):
    if not(videolink == ''):
        cap = cv2.VideoCapture(videolink)

        while True:
            rat,img = cap.read()

            if not rat:
                vehicleCountValue['text'] = '00'
                estimatedTimeValue['text'] = '00'
                videoSourceLabel['text'] = ''
                break

            try: 
                if num == 3:
                    frame = img.copy()
                    frame = frame[350:,500:]
                    blob = cv2.dnn.blobFromImage(frame,1/255.0,(320,320),swapRB=True,crop=False)
                    net.setInput(blob)
                    result = net.forward(ln)
                    idxs,boxes = getBoundingBox(frame,result)
                else:
                    frame = img.copy()
                    frame = cv2.resize(frame,(640,480))
                    blob = cv2.dnn.blobFromImage(frame,1/255.0,(320,320),swapRB=True,crop=False)
                    net.setInput(blob)
                    result = net.forward(ln)
                    idxs,boxes = getBoundingBox(frame,result)

            except:
                frame = np.zeros((480,640))

            if len(idxs)>0:
                count = []
                for j in idxs.flatten():    
                    count.append(j)
                    (x,y) = (boxes[j][0],boxes[j][1])
                    (w,h) = (boxes[j][2],boxes[j][3])

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

                vehicleCountValue['text'] = len(count)
            else:
                vehicleCountValue['text'] = '00'

            if int(vehicleCountValue['text']) > 0:
                mintime = 10
                multipleof = int(vehicleCountValue['text']) // 2
                estimatedTimeValue['text'] = str(mintime + (3*multipleof)) + 'Second'
            

            try:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            except:
                pass

            photo = ImageTk.PhotoImage(Image.fromarray(frame))
            video_label['image'] = photo

            root.update()

    else:
        while videolink == '':
            root.update()



root = Tk()
root.configure(bg='black')
root.state('zoomed')

head_frame = Frame(root,bg='black',borderwidth=5)
head_frame.pack(fill=X,pady=10)
head_label = Label(head_frame,text='Heading',bg='black',fg='Red',font=('','20','bold'))
head_label.pack()

'''Left Frame'''
left_frame = Frame(root,bg='black',borderwidth=5,relief=SUNKEN)
left_frame.pack(fill=Y,side='left')

''' Left Input Frame'''
left_frame_top = Frame(left_frame,bg='black',borderwidth=3,relief=SUNKEN)
left_frame_top.pack(fill=Y,side='left',anchor='n')
# Frame 1: Heading: Input Frames 
subheading1 = Label(left_frame_top,text='Input Cameras',bg='black',fg='White',font=('','20','bold'))
subheading1.pack(padx=100,pady=20)

# Frame1 : Button1: Cam1
cam1_button = Button(left_frame_top,bg='gray',text='CAM1',fg='white',font=('','12','bold'),command=button1)
cam1_button.pack(padx=10,pady=20)

# Frame1 : Button2: Cam2
cam2_button = Button(left_frame_top,bg='gray',text='CAM2',fg='white',font=('','12','bold'),command=button2)
cam2_button.pack(padx=10,pady=20)

# Frame1: Button3 : Frame1
phoframe1_button = Button(left_frame_top,bg='gray',text='CAM3',fg='white',font=('','12','bold'),command=button3)
phoframe1_button.pack(padx=10,pady=20)

# Frame1: Label: HoldsDataNames
dataNameLabel = Label(left_frame_top)

'''Left Output Frame'''
left_frame_bottom = Frame(left_frame,bg='black',borderwidth=3,relief=SUNKEN)
left_frame_bottom.pack(fill=Y,side='left',anchor='s')
# Frame2: Heading: Output Result
subheading2 = Label(left_frame_bottom,text='Output Result',bg='black',fg='white',font=('','20','bold'))
subheading2.pack(padx=100,pady=20)

# Frame2: Padding1
padding1 = Label(left_frame_bottom,pady=15,bg='black')
padding1.pack()

'''# Frame2 : SubFrame1'''
subframe1 = Frame(left_frame_bottom,borderwidth=0,relief=SUNKEN,bg='black')
subframe1.pack(fill=X,pady=10)
# SubFrame1: VehicleCounts
vehicleCountLabel = Label(subframe1,text='  Vehicle Count : ',fg='white',bg='black',font=('','14','bold'))
vehicleCountLabel.pack(side='left',padx=30)
# SubFrame1: Value: VehicleCounts
vehicleCountValue = Label(subframe1,text='00',fg='white',bg='black',font=('','14','bold'))
vehicleCountValue.pack(side='left',padx=0)

'''# Frame2: SubFrame2'''
subframe2 = Frame(left_frame_bottom,borderwidth=0,relief=SUNKEN,bg='black')
subframe2.pack(fill=X,pady=10)
# Subframe2: EstimatedTime 
estimatedTimeLabel = Label(subframe2,text='Estimated Time: ',fg='white',bg='black',font=('','14','bold'))
estimatedTimeLabel.pack(side='left',padx=30)
# Subframe2: Value: EstimatedTime
estimatedTimeValue = Label(subframe2,text='00',fg='white',bg='black',font=('','14','bold'))
estimatedTimeValue.pack(side='left',padx=0)

# Frame2: Padding2
padding2 = Label(left_frame_bottom,pady=100,bg='black')
padding2.pack()

# Frame2: Tilte: Presumption
titleLabel = Label(left_frame_bottom,text='Presumptions',bg='black',fg='white',font=('','14','bold underline'))
titleLabel.pack(pady=10)

'''# Frame2: SubFrame3'''
subframe3 = Frame(left_frame_bottom,borderwidth=0,relief=SUNKEN,bg='black')
subframe3.pack(fill=X,pady=5)
# Subframe3: Minimun Time
minTimeLabel = Label(subframe3,text='Set Minimum Time : ',fg='white',bg='black',font=('','12','bold'))
minTimeLabel.pack(side='left',padx=30)
# Subframe3: Value: Minimun Time
minTimeValue = Label(subframe3,text='10 Seconds',fg='white',bg='black',font=('','12','bold'))
minTimeValue.pack(side='left',padx=0)

'''# Frame2: SubFrame4'''
subframe4 = Frame(left_frame_bottom,borderwidth=0,relief=SUNKEN,bg='black')
subframe4.pack(fill=X,pady=5)
# Subframe4: AverageSpeed
avgSpeedLabel = Label(subframe4,text='Average Vehicle Speed : ',fg='white',bg='black',font=('','12','bold'))
avgSpeedLabel.pack(side='left',padx=30)
# Subframe4: Value: AvgSpeed
avgSpeedValue = Label(subframe4,text='10 Km/h',fg='white',bg='black',font=('','12','bold'))
avgSpeedValue.pack(side='left',padx=0)

'''# Frame2: SubFrame5'''
subframe5 = Frame(left_frame_bottom,borderwidth=0,relief=SUNKEN,bg='black')
subframe5.pack(fill=X,pady=5)
# Subframe5: AvgTime
avgTimeLabel = Label(subframe5,text='Average Time take : ',fg='white',bg='black',font=('','12','bold'))
avgTimeLabel.pack(side='left',padx=30)
# Subframe5: Value: AvgTime
abgTimeValue = Label(subframe5,text='5 Sec/3 Vehicle',fg='white',bg='black',font=('','12','bold'))
abgTimeValue.pack(side='left',padx=0)



'''Right Frame'''
right_frame = Frame(root,bg='black',borderwidth=5,relief=SUNKEN)
right_frame.pack(fill=Y,side='right')

# Frame2: Subheading3: VideoScreen
subheading3 = Label(right_frame,text='Screen',bg='black',fg='White',font=('','20','bold'))
subheading3.pack(padx=310,pady=20)

# Frame2: VideoScreen
video_label = Label(right_frame,bg='black',borderwidth=3)
video_label.pack(padx=40)

# Frame2: Label: VideoSource
videoSourceLabel = Label(right_frame,text='',bg='black',fg='white',font=('','10',''))
videoSourceLabel.pack()




mainScreen()    