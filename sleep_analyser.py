import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from dip import convert_color_channel
import streamlit as st
from PIL import Image

def standardize_image(image):
    image = cv2.resize(image, (700, 500))
    # image = convert_color_channel(image, 'RGB', 'GRAY')
    return image

# Detecting the face to later associate with the facial landmarks
classifier_dblib_68_landmarks = "classifiers/shape_predictor_68_face_landmarks.dat"
classifier_dblib = dlib.shape_predictor(classifier_dblib_68_landmarks)
face_detector = dlib.get_frontal_face_detector()

# Facial landmarks coordinates
FACE = list(range(17, 68))
FULL_FACE = list(range(0, 68))
LIPS = list(range(48, 61))
EYEBROWN_RIGHT = list(range(17, 22))
EYEBROWN_LEFT = list(range(22, 27))
EYE_RIGHT = list(range(36, 42))
EYE_LEFT = list(range(42, 48))
NOSE = list(range(27, 35))
JAW = list(range(0, 17))


def facial_landmarks(image):
    # return faces in 'rectangles'
    rectangles = face_detector(image, 1)
    if len(rectangles) == 0:
        return None

    landmarks  = []
    for rectangle in rectangles:
        landmarks.append(np.matrix([[p.x, p.y] for p in classifier_dblib(image, rectangle).parts()]))
    return landmarks

def draw_landmarks(image, facial_landmarks):
    if image is None:
        return None
    for landmark in facial_landmarks:
        for idx, mark in enumerate(landmark):
            center = (mark[0, 0], mark[0, 1])
            cv2.circle(image, center, 3, color=(255, 255, 0), thickness=-1)
    return image

def mouth_aspect_ratio(mouth_marks):
    a = dist.euclidean(mouth_marks[3], mouth_marks[9])
    b = dist.euclidean(mouth_marks[2], mouth_marks[10])
    c = dist.euclidean(mouth_marks[4], mouth_marks[8])
    d = dist.euclidean(mouth_marks[0], mouth_marks[6])
    aspect_ratio = (a + b + c)/(3.0 * d)
    return aspect_ratio



def run_code():
    video_capture = cv2.VideoCapture(0)
    max_aspect_ratio = 0
    yawn_count = 0
    yawn = False
    previous_yawn = False
    while(True):
        cap, frame = video_capture.read()

        if cap:
            frame = standardize_image(frame)

            marks = facial_landmarks(frame)
            if marks is not None:

                aspect_ratio = mouth_aspect_ratio(marks[0][LIPS])
                aspect_ratio = round(aspect_ratio, 3)

                if aspect_ratio > max_aspect_ratio:
                    max_aspect_ratio = aspect_ratio

                aspect_ratio_info = "ratio aspect " + str(aspect_ratio) + " maximum: " + str(max_aspect_ratio)

                frame = draw_landmarks(frame, marks)

                coord = tuple(marks[0][LIPS][0].A1.reshape(1, -1)[0])
                coord = (coord[0], coord[1] + 20)

                cv2.putText(frame, aspect_ratio_info, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                if aspect_ratio > 1.15:
                    yawn = True
                else:
                    yawn = False
                if previous_yawn == False and yawn == True:
                    yawn_count+=1
                coord = (coord[0], coord[1] + 23)
                cv2.putText(frame, 'yawns:' + str(yawn_count), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                previous_yawn = yawn
            cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Stopping application')
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return yawn_count

# def load_video():
#     uploaded_file = st.file_uploader("Choose a WEBM file", type="webm")
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         image = image.save('images/video.webm')
#         return True
