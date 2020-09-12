#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:36:57 2020
@author: anpandoh

Code created to easily test and manipulate Image Classification Models
"""
########################################################################################################
#Commandline:
#python pythonopener.py --image hhtest1.jpg --config cfg/yolov3-hardhat.cfg --weights backup/yolov3-hardhat_final.weights --names data/obj.names
####################################################################################################
# import required packages
import cv2
import argparse
import numpy as np
import os.path
from os import path

# handle command line arguments
#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
#                help = 'path to input image')
#ap.add_argument('-c', '--config', required=True,
#                help = 'path to config file')
#ap.add_argument('-w', '--weights', required=True,
#                help = 'path to yolo weights')
#ap.add_argument('-n', '--names', required=True,
#                help = 'path to names')
#args = ap.parse_args()


global aa
# pathname = 1
# pathimage =1
# pathcfg =1
# pathweight =1
# image = 1

                
# read input image
def proccessimage ():
    global names
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
   
    
    # read class names from text file
    names = None
    with open(pathname, 'r') as f:
       names = [line.strip() for line in f.readlines()]
    print (names)
    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(names), 3))
    
    
    
    # read pre-trained model and config file
    net = cv2.dnn.readNet(pathweight, pathcfg)
    
    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    # set input blob for the network
    net.setInput(blob)
    
    # function to get the output layer names 
    # in the architecture
    def get_output_layers(net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
        return output_layers
    
    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    
        label = str(names[class_id])
    
        color = COLORS[class_id]
    
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    outs = net.forward(get_output_layers(net))
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.4
    
    # for each detection from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    print(confidences)        
                
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    Hardhat = ()
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        hardhat=(round(x),round(y),round(x+w),round(y+h))
        Hardhat += hardhat
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    print(Hardhat)
    # display output image    
    cv2.imshow("object detection", image)
    
    # wait until any key is pressed
    cv2.waitKey()
        
     # save output image to disk
    cv2.imwrite("object-detection.jpg", image)
    
    # release resources
    cv2.destroyAllWindows()      



# while True:
#         pathimage = input('What is the location of the test image file?\n')
#         image = cv2.imread(pathimage)
#         if (type(image) is np.ndarray):
#             a=1
#             pathname = input('What is the location of the names file?\n')
#             if path.exists(pathname) == True:
#                 pathcfg = input('What is the location of the config file?\n')
#                 if path.exists(pathcfg) == True:
#                     pathweight = input('What is the location of the final weight file?\n')
#                     if path.exists(pathweight) == True:
#                         proccessimage(pathimage, pathname, pathcfg, pathweight)
#                         break
#                     else:     
#                       print("incorrect file path")  
#                 else: 
#                     print("incorrect file path") 
#             else:
#                print("incorrect names file path") 
#         else:
#             print("incorrect file path")



def input1():
    global aa 
    global image
    global pathimage
    if aa==1:
         pathimage = input('What is the location of the test image file?\n')
         image = cv2.imread(pathimage)
         if (type(image) is np.ndarray):
             aa = 2
         else:
             print("incorrect file path")
def input2():
    global aa
    global pathname
    if aa==2:
         pathname = input('What is the location of the names file?\n')
         if path.exists(pathname) == True:
             aa = 3
         else:
             print("incorrect file path")
def input3():
    global aa
    global pathcfg
    if aa == 3:
         pathcfg = input('What is the location of the config file?\n')
         if path.exists(pathcfg) == True:
             aa=4
         else:
             print("incorrect file path")
def input4():
    global aa
    global pathweight
    if aa == 4:
        pathweight = input('What is the location of the final weight file?\n')
        if path.exists(pathweight) == True:
            # proccessimage(pathimage, pathname, pathcfg, pathweight)
            aa = 5
aa = 1
while True:
    input1()
    input2()
    input3()
    input4()
    if aa ==5:
        proccessimage()
        break
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
