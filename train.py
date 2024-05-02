import os
import json
import shutil


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, History
import matplotlib.pyplot as plt

import cv2
import sys




folder_path_video = "D:/arslan/script/outputs/output"
folder_path_audio = "D:/arslan/script/outputs/audio_output"
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
used_counter=0
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
                used_counter=used_counter+1

        except Exception as e:

            continue




    counter=counter+1
    print(used_counter)
    

#print("Final appending array",loaded_arrays_video)
print(np.array(loaded_arrays_video).shape)


#print("Final appending array",loaded_arrays_audio)
print(np.array(loaded_arrays_audio).shape)


print(np.array(label_list).shape)

print(label_list)




combined_features = np.concatenate((loaded_arrays_video, loaded_arrays_audio), axis=2)


x=np.array(combined_features)
y=np.array(label_list)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

# Reshape for LSTM input: (batch_size, time_steps, features)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(3, 10)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Initialize a history object to store training history
history = History()

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model with validation data and save history
history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), callbacks=[history, tensorboard_callback])
model.save('my_model.h5')
# Access training and validation loss from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']  # Use 'val_loss' for validation loss

# Plotting the loss curves
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

