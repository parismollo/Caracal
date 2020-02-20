import cv2
import matplotlib.pyplot as plt
import streamlit as st


def show_image_st(image, *args,**kwargs):
    plt.imshow(image, **kwargs)
    #This seems a very bad implementation, I will find a better later...
    # I'm learning how to use args
    if args:
        plt.title(args[0])
    st.pyplot()

def convert_color_channel(image, f, t):
    if f == 'RGB':
        if t == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif t == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif f == 'BGR':
        if t == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif t == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dip():
    image = cv2.imread('images/image01.png')

    image = convert_color_channel(image, 'BGR', 'RGB')
    image_gray = convert_color_channel(image, 'RGB', 'GRAY')
    image_roi = image[210:400, 500:700]

    show_image_st(image, 'RGB - Image ')
    show_image_st(image_gray,'Gray - Image',cmap='gray')
    show_image_st(image_roi, 'ROI')
