from cv2 import *
import numpy as np
import torch
import json

cam_port = 0
cam = VideoCapture(cam_port) 

result, image = cam.read()      # reading the input using the camera 
print(image.shape)
h,w,_ = image.shape

if result: 
    #imshow("original", image) 
    y = (w-h)//2
    image = image[:, y:y+h]
    image = cv2.resize(image,(224,224))
    print(image.shape)

    imshow("cropped", image) 
    #imwrite("cropped.png", image) 
    
    print(image) 
    print("==================================================================================") 
    
    cuda0 = torch.device('cuda:0')
    rgb_tensor = torch.tensor([image], dtype=torch.uint8, device=cuda0)
    print(rgb_tensor) 
    print("==================================================================================") 
    # waitKey(0) 
    # destroyWindow("original") 
    # destroyWindow("cropped")
    
    batch_dict = {}
    batch_dict['rgb'] = torch.to_json(rgb_tensor)
    batch_json = json.dumps(batch_dict)
    print(batch_json)
    
    
else: 
	print("No image detected. Please! try again") 
