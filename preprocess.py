import os
import json
import shutil
import os
import glob
import gc

import source.audio_analysis_utils.model as audio_model
import source.audio_analysis_utils.predict as audio_predict

import source.face_emotion_utils.model as face_model
import source.face_emotion_utils.predict as face_predict
import source.face_emotion_utils.utils as face_utils
import source.config as config
import source.face_emotion_utils.preprocess_main as face_preprocess_main

import source.audio_face_combined.model as combined_model
import source.audio_face_combined.preprocess_main as combined_data
import source.audio_face_combined.combined_config as combined_config
import source.audio_face_combined.predict as combined_predict
import source.audio_face_combined.download_video as download_youtube
import source.audio_face_combined.utils as combined_utils
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

import cv2
import sys

def calculate_overall_softmax(softmax_lists):
    # Check if softmax_lists is not empty
    if not softmax_lists:
        return [0,0,0,0,0]  # Return None or handle the empty case as needed

    # Sum the softmax scores element-wise
    overall_softmax = np.sum(softmax_lists, axis=0)

    # Normalize the overall softmax scores to add up to 1
    overall_softmax_normalized = overall_softmax / np.sum(overall_softmax)

    return overall_softmax_normalized

# Example usage:
input_folder = "F:/dataset/dfdc_train_all/extracted/dfdc_train_part_1"
output_folder = "G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/data/output"
output_folder2 = "G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/data/audio_output"
buffer_folder = "G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/data/buffer"  # Temporary buffer folder
audio_output="G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/input_files"
complete_files="F:/dataset/dfdc_train_all/extracted/Completed"
un_completed_files="F:/dataset/dfdc_train_all/extracted/uncomplete"

# Function to extract features from a video segment (replace with your custom function)
def extract_features(segment_path, segment_name):
    # Replace this with your feature extraction function
    # Process the segment at 'segment_path' and save the features
    # Make sure to delete the segment file after processing
    # Example: feature_extraction(segment_path, segment_name)
    pass

# Number of segments you want to divide each video into
num_segments = 3
gc.enable()

# Create the buffer folder if it doesn't exist
if not os.path.exists(buffer_folder):
    os.makedirs(buffer_folder)

# List all video files in the input folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
counter = 0
# Loop through each video
for video_file in video_files:

    files = glob.glob('G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/data/buffer/*')
    for f in files:
        os.remove(f)

    

    files = glob.glob('G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/input_files/*')
    for f in files:
        os.remove(f)

    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_length = video_length // num_segments

    all_video_segment_features = []
    all_audio_segment_features = []

    # Process each segment
    for i in range(num_segments):
        segment_name = f"{os.path.splitext(video_file)[0]}_segment_{i+1}.mp4"
        segment_path = os.path.join(buffer_folder, segment_name)

        # Calculate the start and end time for this segment
        start_time = i * segment_length / cap.get(cv2.CAP_PROP_FPS)
        end_time = (i + 1) * segment_length / cap.get(cv2.CAP_PROP_FPS)

        # Extract the video segment using moviepy
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=segment_path)

        video_filename = os.path.splitext(os.path.basename(segment_path))[0]

        # Define the audio file path
        audio_output_path = os.path.join(audio_output, f'{video_filename}.mp3')

        # Extract the audio from the video
        ffmpeg_extract_audio(segment_path, audio_output_path)

        # Pass the segment name and path to your feature extraction function
        
        probability_lists= face_predict.predict(segment_path, video_mode=True)
        final_result=calculate_overall_softmax(probability_lists)
        all_video_segment_features.append(final_result)

        audioprob=audio_predict.predict(video_filename)
        print("audio recieved:\n",audioprob)
        all_audio_segment_features.append(audioprob)


        print("\n\n")
        print(all_video_segment_features)
        # Delete the segment file from the buffer
        

    # Save or use the features extracted from all segments

    if np.array(all_video_segment_features).shape ==(3,5) and np.array(all_audio_segment_features).shape ==(3,5):
        output_filename = os.path.splitext(video_file)[0] + ".npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, np.array(all_video_segment_features))


        
        output_path = os.path.join(output_folder2, output_filename)
        np.save(output_path, np.array(all_audio_segment_features))

        cap.release()
        destination_path=os.path.join(complete_files, video_file)
        shutil.move(video_path, destination_path)
    
    else:
        destination_path=os.path.join(un_completed_files, video_file)
        shutil.move(video_path, destination_path)

    
    counter += 1

    # Run garbage collector every 10 iterations
    if counter == 2:
        gc.collect()
        counter = 0
        os.system('cls')









