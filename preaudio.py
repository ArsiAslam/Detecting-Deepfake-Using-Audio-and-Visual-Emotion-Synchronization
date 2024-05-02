import os
import json
import shutil
import os

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

import cv2
import sys




file_name = str(input("Enter file name: "))
audioprob=audio_predict.predict(file_name)
print("audio recieved:\n",audioprob)