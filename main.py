import streamlit as st
from models.detector import Detector
from models.recognitor import Predictor
from PIL import Image
from utils.generals import draw_bbox_with_text
import matplotlib.pyplot as plt
import numpy as np
import yaml
from copy import deepcopy

__PREDICTOR_CONFIG__ = yaml.load(open("configs/recog/reg.yaml", "rb"), Loader=yaml.Loader)
__DETECTOR_CONFIG__ = yaml.load(open("configs/det/db.yaml", "rb"), Loader=yaml.Loader)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_predictor(config):
    return Predictor(config)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_detector(config):
    return Detector(config)


def main():
    # st.title('OCR')
    # st.subheader('OCR')
    st.markdown('This is a simple ocr app.')
    st.markdown('This app is developed by: Giang Le Truong')
    predictor = get_predictor(__PREDICTOR_CONFIG__)
    detector = get_detector(__DETECTOR_CONFIG__)
    # TODO: upload module
    uploaded_file = st.file_uploader('Upload your picture here!', type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # convert to numpy
        # print(np.array(image).shape)
        bboxes = detector(deepcopy(image))
        images = detector.post_process(deepcopy(image), bboxes)
        sents = predictor(images)
        out_img = draw_bbox_with_text(image, bboxes, sents, color=(0, 0, 255))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(out_img)
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()