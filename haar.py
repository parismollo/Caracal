import cv2
import matplotlib.pyplot as plt
from dip import convert_color_channel, show_image_st
import streamlit as st
from PIL import Image


def haar(image_path):
    face_cascade = cv2.CascadeClassifier('classificators/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('classificators/haarcascade_eye.xml')

    image = cv2.imread(image_path)
    gray = convert_color_channel(image, 'BGR', 'GRAY')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x:x+w]
        roi_color = image[y: y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    image = convert_color_channel(image, 'BGR', 'RGB')
    show_image_st(image, 'Faces and Eyes detection using Haar')

def load_image():
    uploaded_file = st.file_uploader("Choose a JPG file", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.save('images/file.jpg')
        return True
