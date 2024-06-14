import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import pickle

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# if len(tf.config.list_physical_devices('GPU')) > 0:
#     print("Using GPU")
# else:
#     print("No GPU found, using CPU")


# Function to load video and mask, and get parking spot bounding boxes
def load_video_and_mask(video_file, mask_file):
    # Load the video
    cap = cv2.VideoCapture(video_file.name)
    # Load the mask
    mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding boxes for each contour (parking spot)
    parking_spots = [cv2.boundingRect(contour) for contour in contours]
    return cap, parking_spots


# Function to predict occupancy using the trained model
def predict_occupancy(frame, parking_spots, model, target_size):
    predictions = []
    for (x, y, w, h) in parking_spots:
        spot = frame[y:y+h, x:x+w]
        spot_resized = cv2.resize(spot, (model.input_shape[2], model.input_shape[1]))
        spot_resized = np.expand_dims(spot_resized, axis=0)  # Add batch dimension
        prediction = model.predict(spot_resized)
        predictions.append(prediction[0][0])
    return predictions


# Function to process the entire video and export with rectangles and scores
def export_video_with_predictions(video_file, mask_file, output_path, model, input_shape):
    cap, parking_spots = load_video_and_mask(video_file.name, mask_file)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        predictions = predict_occupancy(frame, parking_spots, model, input_shape)

        for i, (x, y, w, h) in enumerate(parking_spots):
            color = (0, 255, 0) if predictions[i] < 0.5 else (0, 0, 255)  # Green for empty, Red for occupied
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{predictions[i]:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()


# Load the trained TensorFlow model
model = pickle.load(open('model_MobileNetV2_4.pkl',"rb"))
input_shape = model.input_shape[1:3]

st.title("Parking Lot Occupancy Detection")

# Upload video file
video_file = st.file_uploader("Choose a video file", type=["mp4"])
mask_file = st.file_uploader("Choose a mask file", type=["png"])

if video_file is not None and mask_file is not None:
    # Load video and mask, and get parking spots
    cap, parking_spots = load_video_and_mask(video_file, mask_file)
    
    # Streamlit widgets for controlling the playback
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Add a slider to select the frame
    frame_number = st.slider("Frame", 0, total_frames - 1, 0)

    # Set video position to the selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if ret:
        # Predict the occupancy for the current frame
        predictions = predict_occupancy(frame, parking_spots, model, input_shape)

        # Draw rectangles and confidence scores on the frame
        for i, (x, y, w, h) in enumerate(parking_spots):
            color = (0, 255, 0) if predictions[i] < 0.5 else (0, 0, 255)  # Green for empty, Red for occupied
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(frame, f'{predictions[i]:.2f}', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert the frame to RGB (Streamlit displays images in RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        st.image(frame_rgb)

    cap.release()

else:
    st.write("Please upload both a video file and a mask file.")
