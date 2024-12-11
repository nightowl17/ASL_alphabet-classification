import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


from util import classify

st.title('ASL alphabet')

st.header('Please upload an image ')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

model = tf.keras.models.load_model('Efficientnet_ASL_alphabet_detection.h5')

with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))



st.text('Note : i used a public dataset to make this project therefore its only alphabets \n for now and the performance may not be satisfactory. \n the classifier requires a clear image of your hand in order to achieve great results. \n this project will be improved to be an object detection project in the future.')