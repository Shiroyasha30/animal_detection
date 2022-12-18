import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
import tensorflow as tf
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt


st.set_page_config(page_title='Animal Detection', layout='wide')


@st.cache
def load_freq():
    freq=pd.read_csv('frequencies.csv')
    return freq


@st.cache
def load_model():
    PATH_TO_SAVED_MODEL="inference_graph/saved_model"
    model=tf.saved_model.load(PATH_TO_SAVED_MODEL)
    placeholder.info("Model Loaded!!")
    time.sleep(1)
    placeholder.empty()
    return model


@st.cache
def load_labels():
    labels=label_map_util.create_category_index_from_labelmap("label_map.pbtxt",use_display_name=True)
    return labels


def page1():
    st.title('Frequency dataset')
    freq=load_freq()
    # print(freq)
    st.dataframe(freq)
    return freq


def page2():
    st.title('Animal Detection using ssd_mobilenet')
    freq=load_freq()
    IMAGE_SIZE = (8, 5)
    image_path= st.file_uploader('Insert image for classification', type=['jpg'])

    detect_fn=load_model()
    
    category_index=load_labels()

    def load_image_into_numpy_array(path):
        im=Image.open(path)
        return im, np.array(im)

    if image_path is not None:
        im, image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.4, # Adjust this value to set the minimum probability boxes to be classified as True
            agnostic_mode=False)

        col1, col2 = st.columns(2)
        with col1:
            st.image(im)

        with col2:
            fig, ax=plt.subplots(figsize=IMAGE_SIZE, dpi=200)
            plt.axis("off")
            plt.imshow(image_np_with_detections)
            plt.show()
            st.pyplot(fig)
            # plt.savefig('out.png')
            # st.image('out.png')

        all_bxs=detections['detection_boxes']
        max_bxs=len(all_bxs)
        scores = detections['detection_scores']
        min_thresh=0.4
        for i in range(max_bxs):
            if scores is None or scores[i] > min_thresh:
                bxs=detections['detection_boxes'][i]
                detected_class=category_index.get(detections['detection_classes'][i])['name']
                st.write('Class : ', detected_class)
                st.write('Frequency to play : ', freq[freq['class']==detected_class]['frequency1'].values[0])
                # print(bxs)
                h, w, c=image_np_with_detections.shape
                xmin = bxs[1]*w
                ymin = bxs[0]*h
                xmax = bxs[3]*w
                ymax = bxs[2]*h
                # print(h, w)
                # print(xmin, ymin, xmax, ymax)
                centroid_x, centroid_y = (xmin+xmax)/2, (ymin+ymax)/2
                # print(centroid_x, centroid_y)
                zone=""
                if centroid_x<w/2:
                    zone+="left "
                else:
                    zone+="right "

                if centroid_y<h/2:
                    zone+="top"
                else:
                    zone+="bottom"

                st.write('Position : ', zone)



placeholder = st.empty()
pages=st.sidebar.selectbox('Section', ['Frequencies', 'Classification'], index=1)
if pages=='Frequencies':
    page1()
elif pages=='Classification':
    page2()