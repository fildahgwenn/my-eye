import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ... (rest of the code)

def capture_image_and_caption():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_caption = generate_caption(caption_processor, caption_model, image)
        st.subheader("Caption:")
        st.write(output_caption)
        generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, output_caption)
        st.subheader("Audio:")
        play_sound()
        with st.expander("See visualization"):
            visualize_speech()
    cap.release()

# ... (rest of the code)

if __name__ == "__main__":
    # ... (rest of the code)

    # Choose image source
    image_source = st.radio("Select Image Source:", ("Upload Image", "Open from URL", "Capture from Camera"))

    if image_source == "Capture from Camera":
        capture_image_and_caption()

    # ... (rest of the code)
