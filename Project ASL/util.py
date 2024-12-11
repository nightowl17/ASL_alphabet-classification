import base64
import tensorflow as tf
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def classify(image, model, class_names):
 
    # Resize image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Preprocess image for EfficientNet
    preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(image_array)

    # Set model input
    data = np.expand_dims(preprocessed_image, axis=0)

    # Make prediction
    prediction = model.predict(data)
    # Choose the index of the highest confidence score
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score