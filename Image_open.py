import streamlit as st
from PIL import Image
from PIL import ImageColor
import numpy as np
from io import BytesIO
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt

def generate_brightness_heatmap(image_np):
    brightness = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(brightness, cmap='YlGnBu', ax=ax, cbar=True)
    ax.set_title('Brightness Heatmap')
    return fig

# Function to generate an RGB heatmap
def generate_rgb_heatmap(image_np, channel_index, cmap):
    # Map the channel index to the color name
    channel_names = ['Red', 'Green', 'Blue']
    channel_name = channel_names[channel_index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(image_np[:, :, channel_index], cmap=cmap, ax=ax, cbar=True)
    ax.set_title(f'{channel_name} Channel Heatmap')
    return fig
# Set the page layout to wide
st.set_page_config(layout="wide", page_title="MISSETECH: Materials Analysis Tool")

model = YOLO('D:/Python/Object Detection Yolo v8/train4/weights/best.pt')
st.markdown(
    """
    <style>
    /* Background and body styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #0b0c10; /* Deep space black */
        color: #c5c6c7; /* Light grey for text */
    }

    /* NASA-themed Headers */
    h1, h2, h3, h4 {
        color: 66fcf1; /* NASA turquoise blue */
        font-family: 'Roboto', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Button styling with futuristic look */
    .stButton>button {
        background-color: #45a29e; /* Teal button color */
        color: white;
        border-radius: 5px;
        padding: 10px 25px;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0px 4px 10px rgba(69, 162, 158, 0.4); /* Soft shadow */
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #66fcf1; /* Brighter blue on hover */
        box-shadow: 0px 6px 15px rgba(102, 252, 241, 0.6); /* Stronger shadow */
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #c5c6c7;
        background-color: #1f2833; /* Darker background for the table */
        color: white; /* White text inside the table */
    }

    /* Image and plot styling */
    .stPlotlyChart, .stImage {
        border-radius: 15px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.5); /* Stronger shadow for depth */
        margin: 20px 0;
    }

    /* Sidebar styling for a NASA control panel feel */
    .sidebar .sidebar-content {
        background-color: #1f2833; /* Space-like grey/black sidebar */
        color: #66fcf1; /* Turquoise blue for sidebar text */
        border-right: 3px solid #45a29e; /* Accent border on the sidebar */
    }

    /* Sidebar headers */
    .sidebar h2 {
        color: #45a29e;
    }

    /* Custom scrollbar in the sidebar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0b0c10; /* Match the body background */
    }

    ::-webkit-scrollbar-thumb {
        background-color: #45a29e; /* NASA teal */
        border-radius: 10px;
        border: 2px solid #0b0c10; /* Outer border matching background */
    }

    /* Accent lines for space-like separation */
   
    hr {
        border: 1px solid 45a29e; /* Thin teal line */
        width: 80%; /* Keep it centered and not too wide */
        margin: 20px auto; /* Add space above and below */
        box-shadow: 0 0 10px rgba(69, 162, 158, 0.5); /* Soft glow effect */
    }


    </style>
    """,
    unsafe_allow_html=True
)

st.title("MISSETECH: Materials Analysis Tool")

with st.sidebar:
    st.subheader("🎨 RGB Color Picker")

    # RGB Input Color Picker
    r_value = st.number_input('R (Red)', min_value=0, max_value=255, value=0, key='r_value')
    g_value = st.number_input('G (Green)', min_value=0, max_value=255, value=0, key='g_value')
    b_value = st.number_input('B (Blue)', min_value=0, max_value=255, value=0, key='b_value')

    # Convert RGB values to HEX
    hex_value = f'#{r_value:02x}{g_value:02x}{b_value:02x}'
    st.markdown(f'<div style="width:335px;height:335px;background-color:{hex_value};"></div>', unsafe_allow_html=True)
    st.write(f"RGB: ({r_value}, {g_value}, {b_value})")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.subheader("Average RGB and Brightness")

    # Create an empty placeholder for the table
    table_placeholder = st.empty()

    # Define an empty DataFrame
    df = pd.DataFrame(columns=["Images", "Average R", "Average G", "Average B", "Brightness"])

    # Display the empty table initially
    table_placeholder.dataframe(df)
    
    st.markdown("<hr>", unsafe_allow_html=True)
# Custom CSS to style the left column
left_col, right_col = st.columns([0.1, 4])

# Left Column: RGB Input Color Picker

   
# The right column will be used for image uploads and display
with right_col:
    st.subheader("🖼️ Image Analysis")
    # Function to display images in 2x2 grid
    def display_images_in_grid(images, captions):
        cols = st.columns(2)  # Create 2 columns
        for i, (img, caption) in enumerate(zip(images, captions)):
            with cols[i % 2]:  # Alternate between columns
                st.image(img, caption=caption, use_column_width=True)

    # Let users upload multiple images
    uploaded_files = st.file_uploader("Choose up to 4 images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    # Process uploaded files
    if uploaded_files is not None:
        
        if len(uploaded_files) > 4:
            st.warning("Please upload exactly 4 images.")
        elif len(uploaded_files) == 4:
            st.success("4 images uploaded successfully!")

            # Initialize lists to store original, detected, and cropped images
            original_images = []
            detected_images = []
            cropped_images = []
            cropped_image_data = []  # Initialize list to store average RGB and brightness data

            # Loop through uploaded images and run YOLO detection
            for i, uploaded_file in enumerate(uploaded_files):
                # Open the image
                img = Image.open(uploaded_file)
                original_images.append(img)  # Store original image

                # Convert image to NumPy array for cropping
                img_np = np.array(img)

                # Save the image to a temporary file so YOLO can process it
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    img.save(temp_file.name)
                    temp_file_path = temp_file.name  # Get the path of the temporary file

                # Run YOLO detection on the temporary file
                results = model(temp_file_path)

                # Remove the temporary file after processing
                os.remove(temp_file_path)

                # Get YOLO detection result and convert to PIL Image
                result_img = Image.fromarray(results[0].plot())
                detected_images.append(result_img)  # Store detected image

                # Crop objects detected by YOLO and calculate RGB/Brightness
                for j, box in enumerate(results[0].boxes.xyxy):
                    x_min, y_min, x_max, y_max = map(int, box)  # Extract bounding box coordinates

                    # Crop the image using the bounding box
                    cropped_img_np = img_np[y_min:y_max, x_min:x_max]
                    cropped_img = Image.fromarray(cropped_img_np)  # Convert NumPy array back to PIL

                    # Add the cropped image to the list
                    cropped_images.append((cropped_img, f"Cropped Object {j+1} from Image {i+1}"))

                    # Calculate average RGB and brightness
                    avg_r = np.mean(cropped_img_np[:, :, 0])
                    avg_g = np.mean(cropped_img_np[:, :, 1])
                    avg_b = np.mean(cropped_img_np[:, :, 2])
                    brightness = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b

                    # Store the data in the list
                    cropped_image_data.append({
                        "Image": f"Image {i+1}",  # Simplified caption
                        "Average R": avg_r,
                        "Average G": avg_g,
                        "Average B": avg_b,
                        "Brightness": brightness
                    })

            # Display original images in 2x2 grid
            st.header("Original Images")
            display_images_in_grid(original_images, [f"Original Image {i+1}" for i in range(4)])

            # Display detected images in 2x2 grid
            st.header("Detected Images")
            display_images_in_grid(detected_images, [f"Detected Image {i+1}" for i in range(4)])

            # Display cropped objects
            if cropped_images:
                st.header("Cropped Objects")
                cropped_imgs, cropped_captions = zip(*cropped_images)
                display_images_in_grid(cropped_imgs, cropped_captions)

            # Update the table in the left column with the new data
            df = pd.DataFrame(cropped_image_data)
            table_placeholder.dataframe(df)

            # Adding the graph part right after table update
with st.sidebar:
    st.subheader("📊 Analysis Results and Graphs")

    if not df.empty:
        # Create a bar chart for average RGB values per image
        st.write("Bar Chart: Average RGB Values")
        fig, ax = plt.subplots()
        bar_width = 0.25
        index = np.arange(len(df))

        ax.bar(index, df['Average R'], bar_width, label='Average R', color='red')
        ax.bar(index + bar_width, df['Average G'], bar_width, label='Average G', color='green')
        ax.bar(index + 2 * bar_width, df['Average B'], bar_width, label='Average B', color='blue')

        ax.set_xlabel('Images')
        ax.set_ylabel('RGB Values')
        ax.set_title('Average RGB Values per Image')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(df['Image'])
        ax.legend()

        # Display the bar chart in Streamlit
        st.pyplot(fig)

        # Create a line chart for brightness values per image
        st.write("Line Chart: Brightness Across Images")
        fig_brightness, ax_brightness = plt.subplots()
        ax_brightness.plot(df['Image'], df['Brightness'], marker='o', color='Purple')

        ax_brightness.set_xlabel('Images')
        ax_brightness.set_ylabel('Brightness')
        ax_brightness.set_title('Brightness Across Images')

        # Display the brightness chart in Streamlit
        st.pyplot(fig_brightness)

        # Add the new RGB values across image plot here
        st.write("Line Chart: RGB Values Across All Pixels for the First Image")
        if len(original_images) > 0:
            # Select the first image for the plot
            img_np = np.array(original_images[0])  # Use the first image uploaded

            # Flatten the image pixels for plotting
            height, width, _ = img_np.shape
            pixels_flat = img_np.reshape(-1, 3)

            x = np.arange(pixels_flat.shape[0])

            # Create the plot for RGB values across the image
            fig_rgb, ax_rgb = plt.subplots(figsize=(10, 6))
            ax_rgb.plot(x, pixels_flat[:, 0], color='red', label='Red', linewidth=0.5)
            ax_rgb.plot(x, pixels_flat[:, 1], color='green', label='Green', linewidth=0.5)
            ax_rgb.plot(x, pixels_flat[:, 2], color='blue', label='Blue', linewidth=0.5)

            ax_rgb.set_title('RGB Values Across the Image')
            ax_rgb.set_xlabel('Pixel Index')
            ax_rgb.set_ylabel('RGB Value')
            ax_rgb.legend()

            # Display the RGB values plot in Streamlit
            st.pyplot(fig_rgb)

            # Generate the heatmaps
            st.subheader(f"Brightness Heatmap for First Image")
            brightness_heatmap = generate_brightness_heatmap(img_np)
            st.pyplot(brightness_heatmap)

            st.subheader(f"RGB Heatmaps for First Image")

            red_heatmap = generate_rgb_heatmap(img_np, 0, 'Reds')
            st.pyplot(red_heatmap)

            green_heatmap = generate_rgb_heatmap(img_np, 1, 'Greens')
            st.pyplot(green_heatmap)

            blue_heatmap = generate_rgb_heatmap(img_np, 2, 'Blues')
            st.pyplot(blue_heatmap)

    else:
        st.write("No data available to plot yet.")
