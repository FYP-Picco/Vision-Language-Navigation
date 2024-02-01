import sys
import pyzed.sl as sl
import numpy as np
import cv2

#initialize camera parameters
init_params = sl.InitParameters()
init_params.set_from_svo_file('Your File')

init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use Ultra depth mode
init_params.coordinate_units = sl.UNIT.METER

rt_param = sl.RuntimeParameters()
rt_param.sensing_mode = sl.SENSING_MODE.FILL

# Create ZED object
zed = sl.Camera()
# Open the SVO file specified as a parameter
err = zed.open(init_params)
#if the video doesn't open successfully
if err != sl.ERROR_CODE.SUCCESS:
    sys.stdout.write(repr(err))
    zed.close()
    exit()
# Prepare container
Depth_image = sl.Mat()
Left_image = sl.Mat()
nb_frames = zed.get_svo_number_of_frames()
while True:
    if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
        #get the frame index
        svo_position = zed.get_svo_position()
        
        zed.retrieve_measure(Depth_image, sl.MEASURE.DEPTH)
        depth_image_rgba = Depth_image.get_data()
        depth_image = cv2.cvtColor(depth_image_rgba, cv2.COLOR_RGBA2RGB)
        cv2.imshow('depth',depth_image)
        
        zed.retrieve_image(Left_image, sl.VIEW.LEFT)
        image_left = Left_image.get_data()
        image_left = cv2.cvtColor(image_left, cv2.COLOR_RGBA2RGB)
        cv2.imshow('image',image_left)
        cv2.waitKey(0)
        if svo_position >= nb_frames-1:  # End of SVO
            break

zed.close()