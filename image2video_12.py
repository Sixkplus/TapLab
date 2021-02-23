#coding:utf-8
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont

# ---------------------------------- Test -----------------------------------------

# PATH_FOLDERS= ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt',\
#             'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld',\
#             'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']

# ---------------------------------- Val ------------------------------------------
PATH_FOLDERS= ['frankfurt', 'lindau', 'munster']

output_shape = (2048,1024)


clip_size = 30
#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')


for PATH_IMGS in PATH_FOLDERS:
    jpgNames = os.listdir(PATH_IMGS)

    i = 0
    while(i < len(jpgNames)):
        if(i % clip_size == 0):
            video_writer = cv2.VideoWriter(filename=jpgNames[i+19].split('.png')[0]+ '.avi', fourcc=fourcc, fps=17, frameSize=output_shape)
            # from 9 -> 20, 12 frames
            i += 8
        curImgName = jpgNames[i]
        curImgPath = os.path.join(PATH_IMGS, curImgName)

        img = cv2.imread(filename=curImgPath)
        img = cv2.resize(img, output_shape, interpolation = cv2.INTER_LINEAR)
        cv2.waitKey(100)
        video_writer.write(img)
        
        if(i % 30 == 20):
            print(curImgName + ' done!')
            video_writer.release()
            i += 9
        i += 1


