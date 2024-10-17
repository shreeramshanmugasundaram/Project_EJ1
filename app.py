import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import tempfile
import os
import shutil
from actions.videoToFrame import runVideoToFrame
from actions.segmentation import runSegmentation
from actions.frameSelection import runFrameSegmentation
from actions.ejectionFraction import runEjectionFraciton

import streamlit as st
import time
import hydralit_components as hc

SAVE_PATH = "test/inputs"


st.markdown(f"<h1>Cardic Health Monitoring using Deep Learning - <br/><br/>Ejection Fraction</h1>", unsafe_allow_html=True)
# Upload videos
video_file1 = st.file_uploader("Upload 2 Chamber Video", type=["mp4", "avi", "mov"])
video_file2 = st.file_uploader("Upload 4 Chamber Video", type=["mp4", "avi", "mov"])

# Input watermark text
gender_text = st.radio(
    "Enter Gender",
    ('male', 'female')
)


def clear_folder(folder_path):
    shutil.rmtree(folder_path)  # Remove the folder and all its contents
    os.mkdir(folder_path)  # Recreate the empty folder

# Usage

if st.button("Submit"):
    if video_file1 and video_file2 and gender_text :
        clear_folder(SAVE_PATH)
        saved_video1_path = os.path.join(SAVE_PATH, "2ch_2.avi")
        saved_video2_path = os.path.join(SAVE_PATH, "4ch_2.avi")

        # Save uploaded videos to the defined folder
        with open(saved_video1_path, "wb") as f1, open(saved_video2_path, "wb") as f2:
            f1.write(video_file1.read())
            f2.write(video_file2.read())
            runVideoToFrame()
            st.write("Converted video to frame")
            st.write("startred segmentataion")


            # Use Hydralit loader
            with hc.HyLoader('', hc.Loaders.pulse_bars):
                runSegmentation()
            st.write("segmentataion completed")
            st.write("selecting frame")
            runFrameSegmentation()
            st.write("Frame Selected")
            st.write("calculating Ejection Fraction")

            output = runEjectionFraciton(gender_text)
 

            # Provide download links
            st.write("Watermarked videos are ready for download.")

            st.markdown(f"<b>{output}</b>", unsafe_allow_html=True)
            # with open(output_video1_path, "rb") as file1:
            #     st.download_button(label="Download First Watermarked Video", data=file1, file_name="watermarked_video1.avi", mime="video/x-msvideo")

            # with open(output_video2_path, "rb") as file2:
            #     st.download_button(label="Download Second Watermarked Video", data=file2, file_name="watermarked_video2.avi", mime="video/x-msvideo")
