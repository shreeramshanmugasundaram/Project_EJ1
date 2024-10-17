import cv2
import os


def convert_video_to_frames(input_video_path, output_folder, num_frames=150):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame interval to get exactly num_frames frames
    frame_interval = max(1, total_frames // num_frames)

    # Initialize a counter for naming the frames and a counter for saved frames
    frame_counter = 0
    saved_frames = 0

    # Loop through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save frames at the calculated interval
        if frame_counter % frame_interval == 0:
            # Save the frame as an image
            frame_path = os.path.join(output_folder, f'frame_{saved_frames:04d}.jpg')
            cv2.imwrite(frame_path, frame)

            saved_frames += 1

            # Break if we have saved num_frames frames
            if saved_frames == num_frames:
                break

        # Increment the frame counter
        frame_counter += 1

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Usage
input_video_path_2ch = 'test/inputs/2ch_2.avi'
output_folder_2ch= 'test/outputs/frames/2'
input_video_path_4ch = 'test/inputs/4ch_2.avi'
output_folder_4ch= 'test/outputs/frames/4'
num_frames = 150

def runVideoToFrame():
    convert_video_to_frames(input_video_path_2ch, output_folder_2ch, num_frames)
    convert_video_to_frames(input_video_path_4ch, output_folder_4ch, num_frames)