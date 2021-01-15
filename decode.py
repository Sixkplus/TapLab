# The script aims to obtain the compressed information
# Visualization the motion vectors and residual graph

import numpy as np 
import cv2
import os, glob
import time
from skimage.io import imread, imsave

from coviar import load
from coviar import get_num_frames

GOP_FRAMES_NUM = 12

# The dir for RGB frames
PATH_RGB_FRAMES = 'rgb_cont'

# The dir for continuous MV
PATH_MV_CONT = 'mv_cont/'
# The dir for Residual map
PATH_RES_CONT = 'res_cont/'


video_names = glob.glob('./*.avi')
video_names.sort()


def main():
    #for video_name in video_names[:2]:
    for video_name in video_names:
        fold_path = video_name.split('.avi')[0].split('/')[-1]
        path_img = os.path.join(fold_path, PATH_RGB_FRAMES)
        path_mv = os.path.join(fold_path, PATH_MV_CONT)
        path_res = os.path.join(fold_path, PATH_RES_CONT)
        if not os.path.exists(path_img):
            os.makedirs(path_img)
        if not os.path.exists(path_mv):
            os.makedirs(path_mv)
        if not os.path.exists(path_res):
            os.makedirs(path_res)
        NUM_FRAMES = get_num_frames(video_name)
        print(NUM_FRAMES)
        # The index of GOP
        curGopIdx = 0
        for curGopIdx in range(max(NUM_FRAMES // GOP_FRAMES_NUM, 1)):
            for innerGopIdx in range(GOP_FRAMES_NUM):
                curFrameIdx = curGopIdx * GOP_FRAMES_NUM + innerGopIdx
                print(video_name, curGopIdx, innerGopIdx)
                
                # start = time.time()
                rgbFrame = load(video_name, curGopIdx, innerGopIdx, 0, True)
                # print("rgb time: ", time.time()-start)
                cv2.imwrite(path_img+'/frame'+str(curFrameIdx)+'.png', rgbFrame)
                
                # start = time.time()
                mvCont_origin = load(video_name, curGopIdx, innerGopIdx, 1, False)
                # print("mv time: ", time.time()-start)

                # start = time.time()
                resCont = load(video_name, curGopIdx, innerGopIdx, 2, False)
                # print("res time: ", time.time()-start)

                if mvCont_origin is None:
                    mvCont_origin = np.zeros([1024,2048,2], dtype=np.uint8)
                
                mvCont = mvCont_origin + 2048
                # (high_h, low_h, high_w, low_w)
                mvPng = np.array([((mvCont[:,:,0] >> 8) & 0xff) , (mvCont[:,:,0] & 0xff), ((mvCont[:,:,1] >> 8) & 0xff), (mvCont[:,:,1] & 0xff)], dtype = np.uint8)
                mvPng = np.transpose(mvPng, [1,2,0])
                
                imsave(path_mv+'/frame'+str(curFrameIdx)+'.png', mvPng)

                if resCont is None:
                    resCont = np.zeros([1024,2048,3], dtype=np.uint8)
                
                resCont = np.round((resCont + 256)/2).astype(np.uint8)
                imsave(path_res+'/frame'+str(curFrameIdx)+'.png', resCont)
                

if __name__ == "__main__":
    main()