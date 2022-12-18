import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

model= torch.hub.load('ultralytics/yolov5','custom',path='weights\last.pt')

def predict(img):
    results=model(img)
    text=results.pandas().xyxy[0]
    text=text['name']
    present=False
    if 'weed 1' in text or 'weed 2' in text or 'weed 3' in text:
        present=True     
    box=np.squeeze(results.render())
    return box,present

img=cv2.imread(IMG_PATH)
box,present=predict(img)
if present:
    print("Weed is present")
plt.imshow()
plt.show()

