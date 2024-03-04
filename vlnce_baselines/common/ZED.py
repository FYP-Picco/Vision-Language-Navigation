import cv2
import pyzed.sl as sl
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
# init_params.camera_fps = 15

# Set sensing mode in FILL
runtime_parameters =sl.RuntimeParameters()
sl.RuntimeParameters.enable_fill_mode
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.15  # Set the minimum depth perception distance to 15cm
init_params.depth_maximum_distance = 20      # Set the maximum depth perception distance to 20m
runtime_parameters = sl.RuntimeParameters()

cuda0 = torch.device('cuda:0')

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
  print("Camera Failed")
  exit(-1)
else: 
  print("Camera Initiated Successfully")

class Cam():
    
  def zedInit():
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
      print("Camera Failed")
      exit(-1)
    else: 
      print("Camera Initiated Successfully")
    
  def showImages(rgb, depth,corrected_depth):
    rgb = cv2.resize(rgb,(256,256))
    
    cv2.imshow("img", rgb) #Display image
    cv2.moveWindow('img',30,100)
    cv2.imshow("dep", depth) #Display image
    cv2.moveWindow('dep',420,100)
    cv2.imshow("corr_dep", corrected_depth) #Display image
    cv2.moveWindow('corr_dep',820,100)
      
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

  def getRGB():
    image = sl.Mat()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
      zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve the left image
      rgb = image.get_data()      
      h,w,_ = rgb.shape
      y = (w-h)//2
      rgb = rgb[:, y:y+h]
      rgb = cv2.resize(rgb,(224,224))      
      rgb=rgb[:,:,:3]
      return rgb
    else:
      print("Failed to get RGB image")
      return
    
  def adjust_gamma(image, gamma=1):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # plt.plot(table)
    # plt.show()
    return cv2.LUT(image, table)


  def getDepth():
    depth = sl.Mat()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
      zed.retrieve_image(depth, sl.VIEW.DEPTH) # Retrieve depth image
      depth = depth.get_data()
      h,w,_ = depth.shape
      y = (w-h)//2
      depth = depth[:, y:y+h]
      depth = cv2.resize(depth,(256,256))
      
      corrected_depth = Cam.adjust_gamma(depth,0.35) 
      # corrected_depth = cv2.GaussianBlur(corrected_depth,(3,3),cv2.BORDER_DEFAULT)  
      corrected_depth=np.array([[[col[0]] for col in row]for row in corrected_depth])  
      diff = np.max(corrected_depth)-np.min(corrected_depth) 
      corrected_depth=np.array([[[1-round(col[0]/diff,4)] for col in row]for row in corrected_depth])    
      
      depth=np.array([[[col[0]] for col in row]for row in depth])  
      diff = np.max(depth)-np.min(depth) 
      depth=np.array([[[1-round(col[0]/diff,4)] for col in row]for row in depth])    
      
      return depth,corrected_depth
    else:
      print("Failed to get depth image")
      return
    
  def getRGB_t(rgb):
    return torch.tensor([rgb], device=cuda0)
  
  def getDepth_t(depth):
    return torch.tensor([depth], dtype=torch.float, device=cuda0)
  
  def newFrame(): 
    rgbArray = Cam.getRGB()
    depthArray, corrected_depthArray = Cam.getDepth()
    Cam.showImages(rgbArray, depthArray,corrected_depthArray)  
    rgb = Cam.getRGB_t(rgbArray)
    depth = Cam.getDepth_t(depthArray) 
    corr_depth = Cam.getDepth_t(corrected_depthArray)
    return rgb, depth, corr_depth
    

  


# rgb,depth,correctedDepth= Cam.newFrame()
# Cam.showImages(rgb,depth, correctedDepth)


  

