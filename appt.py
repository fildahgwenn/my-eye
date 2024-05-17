import cv2
import streamlit as st
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Model Description
model_description = """
This application utilizes image captioning and text-to-speech models to generate a caption for an uploaded image 
and convert the caption into speech.

The image captioning model is based on [Salesforce's BLIP architecture](https://huggingface.co/Salesforce/blip-image-captioning-base), which can generate descriptive captions for images.

The text-to-speech model, based on [Microsoft's SpeechT5](https://huggingface.co/microsoft/speecht5_tts), converts the generated caption into speech with the help of a 
HiFiGAN vocoder.
"""

# Initialize Streamlit page configuration
def initialize_page():
    st.set_page_config(
        page_title="Assistive app for the visually impaired",
        page_icon="👁️‍🗨️",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

# Function to capture live video stream and caption it
def capture_and_caption_video():
    cap = cv2.VideoCapture(0)

    # Initialize image captioning models
    caption_processor, caption_model = initialize_image_captioning()

    # Initialize speech synthesis models
    speech_processor, speech_model, speech_vocoder, speaker_embeddings = initialize_speech_synthesis()

    while True:
        ret, frame = cap.read()

        # Display the captured video stream
        cv2.imshow('Live Video Stream', frame)

        # Generate caption for each frame
        output_caption = generate_caption(caption_processor, caption_model, Image.fromarray(frame))

        # Generate speech from the caption
        generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, output_caption)

        # Press 'q' to exit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    initialize_page()

    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title'>Assistive app for the visually impaired</div>", unsafe_allow_html=True)

    # Button to start live video stream captioning
    if st.button("Start Live Video Captioning"):
        capture_and_caption_video()

if __name__ == "__main__":
    main()
