# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:28:11 2024

@author: abner
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2
import base64
from io import BytesIO

# Load your trained YOLO model
model = YOLO('D:/Python/Object Detection Yolo v8/train4/weights/best.pt')

# Function to display images in Streamlit
def display_image_with_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    img_html = f'<img src="data:image/png;base64,{img_base64}" style="width:400px; height:300px;">'
    st.markdown(img_html, unsafe_allow_html=True)

# Function to plot RGB values
def plot_rgb_values(img_np):
    height, width, _ = img_np.shape
    pixels_flat = img_np.reshape(-1, 3)
    x = np.arange(pixels_flat.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, pixels_flat[:, 0], color='red', label='Red', linewidth=0.5)
    ax.plot(x, pixels_flat[:, 1], color='green', label='Green', linewidth=0.5)
    ax.plot(x, pixels_flat[:, 2], color='blue', label='Blue', linewidth=0.5)
    ax.set_title('RGB Values Across the Image')
    ax.set_xlabel('Pixel Index')
    ax.set_ylabel('RGB Value')
    ax.legend()
    st.pyplot(fig)

# Main App
st.title("YOLO Object Detection and Cropping")

# Upload Images
uploaded_files = st.file_uploader("Choose Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"Image {i+1}: {uploaded_file.name}")

        # Load image
        img = Image.open(uploaded_file)
        
        # Display the original image
        st.subheader(f"Original Image {i+1}")
        display_image_with_base64(img)

        # Run YOLO detection
        results = model(uploaded_file)
        
        # Plot and display detection results
        result_img = Image.fromarray(results[0].plot())
        st.subheader(f"Detected Objects in Image {i+1}")
        display_image_with_base64(result_img)

        # Crop detected objects and display them
        img_np = np.array(img)
        for j, result in enumerate(results[0].boxes.xyxy):
            x_min, y_min, x_max, y_max = map(int, result)
            cropped_img = img_np[y_min:y_max, x_min:x_max]
            cropped_pil = Image.fromarray(cropped_img)
            st.subheader(f"Cropped Object {j+1} from Image {i+1}")
            display_image_with_base64(cropped_pil)

    # Analyze an example image for RGB values
    st.subheader("RGB Analysis of a Sample Image")
    sample_image_path = 'D:/MISSE 11 images example/LARC_MIR-1_1550050619B15017180-defish.jpg'
    sample_img = Image.open(sample_image_path)
    sample_img_np = np.array(sample_img)
    plot_rgb_values(sample_img_np)