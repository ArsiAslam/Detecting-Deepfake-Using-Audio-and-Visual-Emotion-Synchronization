import os
import json
import shutil
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

import cv2
import sys
from keras.models import load_model




folder_path_video = "D:/arslan/testing/script/data/output"
folder_path_audio = "D:/arslan/testing/script/data/audio_output"
loaded_arrays_video = []
loaded_arrays_audio = []
label_list = []
metadata_file_path = 'G:/My Drive/Thesis 1 (Deep Fake Detection)/Thesis-II/Codes/Pre-Processed DFDC/metadata.json'

with open(metadata_file_path, 'r') as f:
    metadata = json.load(f)


'''
# Read metadata
with open(metadata_file_path, 'r') as f:
    metadata = json.load(f)

# Initialize lists to store data and labels



# Iterate through each entry in metadata
for file_name, file_info in metadata.items():
    # Remove '.mp4' extension from the file name
    video_name = file_name.replace('.mp4', '')
    file_path = os.path.join(folder_path_video, video_name + '.npy')
    file_path_audio = os.path.join(folder_path_audio, video_name + '.npy')
    
    # Check if the numpy file exists
    if os.path.exists(file_path):
        # Load numpy file
        numpy_data_video = np.load(file_path, allow_pickle=True)
        numpy_data_audio = np.load(file_path_audio, allow_pickle=True)
        
        # Append data to the list
        loaded_arrays_video.append(numpy_data_video)
        loaded_arrays_audio.append(numpy_data_audio)

        
        # Extract label and convert to binary (0 for REAL, 1 for FAKE)
        label = 1 if file_info['label'] == 'FAKE' else 0
        label_list.append(label)
    else:
        print(f"Warning: Numpy file not found for video {video_name}. Skipping.")

# Convert lists to numpy arrays
X = np.array(loaded_arrays_video)
y = np.array(label_list)
z = np.array(loaded_arrays_audio)
# Print shapes of the arrays
print("Shape of data array:", X.shape)
print("Shape of label array:", y.shape)
print("Shape of audio array:", z.shape)






'''

counter=0
for filename in os.listdir(folder_path_video):
    if filename.endswith(".npy"):
        print(filename)


        print(os.path.join(folder_path_video, filename))

        try:

            array = np.load(os.path.join(folder_path_video, filename),allow_pickle=True)
            #print(array.shape)

            array2 = np.load(os.path.join(folder_path_audio, filename),allow_pickle=True)
            
            if array.shape == (3, 5):
                loaded_arrays_video.append(array)
                loaded_arrays_audio.append(array2)
                #print("name",os.path.join(folder_path, filename) )
                #print("Printing array:",array)
                video_name = os.path.splitext(filename)[0] + '.mp4'
                label = metadata[video_name]['label']
                # Convert label to binary (0 for REAL, 1 for FAKE)
                binary_label = 0 if label == 'REAL' else 1
                label_list.append(binary_label)

        except Exception as e:

             pass    



    counter=counter+1
    print(counter)
    #if counter==200:
        

#print("Final appending array",loaded_arrays_video)
print(np.array(loaded_arrays_video).shape)

#print(loaded_arrays_video)


#print("Final appending array",loaded_arrays_audio)
print(np.array(loaded_arrays_audio).shape)


print(np.array(label_list).shape)

print(label_list)




combined_features = np.concatenate((loaded_arrays_video, loaded_arrays_audio), axis=2)



x=np.array(combined_features)
y=np.array(label_list)


loaded_model = load_model("my_model.h5")
evaluation_result = loaded_model.evaluate(x, y)
print("Evaluation Result:", evaluation_result)

predictions = loaded_model.predict(x)
rounded_predictions = np.round(predictions)

# Compute confusion matrix
conf_matrix = confusion_matrix(y, rounded_predictions)

# Display the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Not Class 1", "Class 1"], 
            yticklabels=["Not Class 1", "Class 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

