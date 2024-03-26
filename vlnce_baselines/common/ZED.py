import cv2
import pyzed.sl as sl
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

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
  
  def __init__(self):
    self.processor = AutoImageProcessor.from_pretrained("nielsr/depth-anything-small")
    self.model = AutoModelForDepthEstimation.from_pretrained("nielsr/depth-anything-small")
    
  def zedInit():
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
      print("Camera Failed")
      exit(-1)
    else: 
      print("Camera Initiated Successfully")
    
  def showImages(self,rgb,depth,corrected_depth,adjusted,depth_zed, corrected_depth_zed):
    # rgb = cv2.resize(rgb,(256,256))
    
    cv2.imshow("img", rgb) #Display image
    cv2.moveWindow('img',30,100)
    # cv2.imshow("sharpened img", sharpened) #Display image
    # cv2.moveWindow('sharpened img',420,100)
    cv2.imshow("adjusted img", adjusted) #Display image
    cv2.moveWindow('adjusted img',420,100)
    cv2.imshow("dep", depth) #Display image
    cv2.moveWindow('dep',820,100)
    cv2.imshow("corr_dep", corrected_depth) #Display image
    cv2.moveWindow('corr_dep',1220,100)
    cv2.imshow("zed_org", depth_zed) #Display image
    cv2.moveWindow('zed_org',1620,100)
    cv2.imshow("zed_org_corr", corrected_depth_zed) #Display image
    cv2.moveWindow('zed_org_corr',30,500)
    # =================================================
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

  def closeAllWindows(self):    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

  def sharpen_image(self,image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
  
  def getRGB(self):
    image = sl.Mat()
    alpha = 1.5
    beta = 1.2

    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
      zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve the left image
      rgb = image.get_data()      
      h,w,_ = rgb.shape
      y = (w-h)//2
      rgb = rgb[:, y:y+h]
      
      sharpened_org = self.sharpen_image(rgb)
      sharpened_org = sharpened_org[:,:,:3]
      adjusted_image_org = cv2.convertScaleAbs(sharpened_org, alpha, beta) 
      
      sharpened = cv2.resize(sharpened_org,(224,224))     
      sharpened=sharpened[:,:,:3]
      adjusted_image = cv2.convertScaleAbs(sharpened, alpha, beta)
      
      rgb = cv2.resize(rgb,(224,224))      
      rgb=rgb[:,:,:3]
      
      return rgb,adjusted_image_org,adjusted_image
    else:
      print("Failed to get RGB image")
      return
    
  def adjust_gamma(self,image, gamma=1):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # plt.plot(table)
    # plt.show()
    return cv2.LUT(image, table)

  def getDepth(self):
    depth = sl.Mat()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
      zed.retrieve_image(depth, sl.VIEW.DEPTH) # Retrieve depth image
      depth = depth.get_data()
      h,w,_ = depth.shape
      y = (w-h)//2
      depth = depth[:, y:y+h]
      depth = cv2.resize(depth,(256,256))
      
      corrected_depth = self.adjust_gamma(depth,0.35) 
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
    
  def get_depthAny(self,image):
    
    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
      outputs = self.model(pixel_values)
      predicted_depth = outputs.predicted_depth
      
    h, w = 224, 224

    depth = torch.nn.functional.interpolate(predicted_depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)

    # Round the values to 4 decimal places
    corrected_depth = self.adjust_gamma(depth,0.35)
    corrected_depth = cv2.resize(corrected_depth,(256,256))
    # corrected_depth=np.array([[[col] for col in row]for row in corrected_depth])
    corrected_depth = np.array([[[1-np.round(col/255,4)] for col in row]for row in corrected_depth])
    
    # corrected_depth_3d = np.stack([corrected_depth] * 3, axis=2)
    
    dep = cv2.resize(depth,(256,256))
    dep = np.array([[[1-np.round(col/255,4)] for col in row]for row in dep])
    
    # dep_3d = np.stack([dep] * 3, axis=2)
    return dep, corrected_depth

  def getRGB_t(self,rgb):
    return torch.tensor([rgb], device=cuda0)
  
  def getDepth_t(self,depth):
    return torch.tensor([depth], dtype=torch.float, device=cuda0)
  
  def newFrame(self): 
    rgbArray,adjusted_image_orgArray,adjustedArray = self.getRGB()
    depthArray, corrected_depthArray= self.get_depthAny(adjusted_image_orgArray)
    depthArray_zed, corrected_depthArray_zed = self.getDepth()
    self.showImages(rgbArray,depthArray,corrected_depthArray,adjustedArray,depthArray_zed, corrected_depthArray_zed)  
    
    self.closeAllWindows()
    # rgb = self.getRGB_t(rgbArray)
    adjusted = self.getRGB_t(adjustedArray)
    # depth = self.getDepth_t(depthArray) 
    corr_depth = self.getDepth_t(corrected_depthArray)
    # sharpened_rgb = self.getRGB_t(sharpenedArray)
    return corr_depth,adjusted
    

  


# cam_instance = Cam()  # Instantiate the Cam class
# correctedDepth, sharpened_rgb = cam_instance.newFrame()  # Call the newFrame() method on the cam_instance
# Cam.showImages(rgb,depth, correctedDepth)


  

