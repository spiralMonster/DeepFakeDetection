import cv2 as cv
import numpy as np
from format_frames import FormatFrames

# frame_data_shape=[frame,width,heigth,channel]

def GetOpticalFlowFrames(frame_data):
    result=[]
    frame_width=frame_data.shape[1]
    frame_height=frame_data.shape[2]

    for ind in range(frame_data.shape[0]-1):
        old_frame=frame_data[ind]
        curr_frame=frame_data[ind+1]

        #Converting frames into GrayScale
        old_frame=cv.cvtColor(old_frame,cv.COLOR_RGB2GRAY)
        curr_frame=cv.cvtColor(curr_frame,cv.COLOR_RGB2GRAY)

        #Computing Optical Flow:
        flow=cv.calcOpticalFlowFarneback(old_frame,curr_frame,None,0.5, 3, 15, 3, 5, 1.2, 0)

        #Compute polar coordinates:
        mag,ang=cv.cartToPolar(flow[...,0],flow[...,1])

        #Compute HSV image:
        hsv_img=np.zeros(shape=(frame_width,frame_height,3))
        hsv_img[...,0]=ang*180/np.pi/2
        hsv_img[...,1]=255
        hsv_img[...,2]=cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

        #Convert HSV to RGB image:
        hsv_img=np.unit8(hsv_img)
        optical_flow_frame=cv.cvtColor(hsv_img,cv.COLOR_HSV2RGB)
        result.append(FormatFrames(frame=optical_flow_frame,output_shape=[frame_width,frame_height]))

    result=np.array(result)
    result=np.expand_dims(result,axis=0)
    return result


