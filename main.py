import streamlit as st
from dip import dip
from haar import haar, load_image
from sleep_analyser import run_code
import cv2
def main():
    st.sidebar.title("Computer Vision")
    app_mode = st.sidebar.selectbox("Choose the app mode",
            ["DIP", "Haar", 'Sleep Analyser'])
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
            
    if app_mode == "Sleep Analyser":
        st.sidebar.success('Running Sleep Analyser mode')
        if st.button('Press the button to start'):
            yawn_count = run_code()
            st.success(f'Yawns counted: {yawn_count}')
            if yawn_count > 2:
                st.title('GO SLEEP!')
        else:
            st.markdown('**Press Q to stop anytime.**')
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
