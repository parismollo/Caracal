import streamlit as st
from dip import dip
from haar import haar, load_image
def main():
    st.sidebar.title("Computer Vision")
    app_mode = st.sidebar.selectbox("Choose the app mode",
            ["DIP", "Haar"])
    if app_mode == "DIP":
        st.sidebar.success('Running Digital Image Processing mode')
        dip()
    if app_mode == "Haar":
        st.sidebar.success('Running Haar')
        haar('images/px-people.jpg')
        st.subheader('Try it yourself')
        st.markdown('Upload a JPG image of your face and see if it works!')
        path = load_image()
        if path:
            haar('images/file.jpg')

if __name__ == '__main__':
    main()
