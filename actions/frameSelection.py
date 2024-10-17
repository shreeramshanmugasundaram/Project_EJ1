import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt

# Function to select end-systolic and end-diastolic frames using frame differencing
def select_frames(input_folder_2ch, input_folder_4ch, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process 2ch frames
    process_frames(input_folder_2ch, os.path.join(output_folder, "2ch"))

    # Process 4ch frames
    process_frames(input_folder_4ch, os.path.join(output_folder, "4ch"))

# Function to process frames in a given input folder
def process_frames(input_folder, output_subfolder):
    # Create the output subfolder
    os.makedirs(output_subfolder, exist_ok=True)

    # Get the list of mask files sorted by modification time
    mask_files = sorted(os.listdir(input_folder), key=lambda f: os.path.getmtime(os.path.join(input_folder, f)))
    num_frames = len(mask_files)

    # Calculate frame differences
    frame_diffs = []
    for i in range(1, num_frames):
        prev_frame = cv2.imread(os.path.join(input_folder, mask_files[i-1]), cv2.IMREAD_GRAYSCALE)
        curr_frame = cv2.imread(os.path.join(input_folder, mask_files[i]), cv2.IMREAD_GRAYSCALE)
        diff = np.mean(np.abs(prev_frame - curr_frame))
        frame_diffs.append(diff)

    # Find indices of frames with maximum and minimum differences
    end_systolic_frame_idx = np.argmax(frame_diffs)
    end_diastolic_frame_idx = np.argmin(frame_diffs)

    # Copy end-systolic and end-diastolic frames to the output folder
    shutil.copy2(os.path.join(input_folder, mask_files[end_systolic_frame_idx]), os.path.join(output_subfolder, f'end_systolic_frame.png'))
    shutil.copy2(os.path.join(input_folder, mask_files[end_diastolic_frame_idx]), os.path.join(output_subfolder, f'end_diastolic_frame.png'))


# Define input and output folders
input_folder_2ch = "test/outputs/segmented/2ch"
input_folder_4ch = "test/outputs/segmented/4ch"
output_folder = "test/outputs/main"

# Select end-systolic and end-diastolic frames and save them into output folders
# select_frames(input_folder_2ch, input_folder_4ch, output_folder)

def runFrameSegmentation():
    select_frames(input_folder_2ch, input_folder_4ch, output_folder)



