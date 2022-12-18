import streamlit as st
import torch
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
# import cv2
import os

@st.cache
def load_model():
    mod=torch.hub.load('ultralytics/yolov5','custom',path='weights\last.pt', force_reload=True)
    return mod

def func1():
    model= load_model()

    IMG_PATH= st.file_uploader('Insert image for classification', type=['jpg'])
    def predict(img):
        results=model(img)
        text=results.pandas().xyxy[0]
        text=text['name']
        present=False
        if 'weed 1' in text or 'weed 2' in text or 'weed 3' in text:
            present=True     
        box=np.squeeze(results.render())
        return box,present

    if IMG_PATH is not None:
        im=Image.open(IMG_PATH)
        img=plt.imread(IMG_PATH)
        box,present=predict(img)
        if present:
            st.write("Weed is present")
        plt.imshow()
        plt.show()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(im)

        with col2:
            IMAGE_SIZE = (8, 5)
            fig, ax=plt.subplots(figsize=IMAGE_SIZE, dpi=200)
            plt.axis("off")
            plt.imshow()
            plt.show()
            st.pyplot(fig)

func1()

# import torch
# from matplotlib import pyplot as plt
# import numpy as np
# import cv2
# import os

# model= torch.hub.load('ultralytics/yolov5','custom',path='weights\last.pt')

# def predict(img):
#     results=model(img)
#     text=results.pandas().xyxy[0]
#     text=text['name']
#     present=False
#     if 'weed 1' in text or 'weed 2' in text or 'weed 3' in text:
#         present=True     
#     box=np.squeeze(results.render())
#     return box,present

# img=cv2.imread(IMG_PATH)
# box,present=predict(img)
# if present:
#     print("Weed is present")
# plt.imshow()
# plt.show()

