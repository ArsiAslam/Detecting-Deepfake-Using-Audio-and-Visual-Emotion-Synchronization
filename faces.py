import cv2
import numpy as np
import os

def count_people_yolo(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    test=0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0 corresponds to person class in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                test+=1
             
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    
    return len(indices)

def process_videos_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {input_video_path}")
            cap = cv2.VideoCapture(input_video_path)

            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            layer_names = net.getUnconnectedOutLayersNames()

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                num_people = count_people_yolo(frame, net, layer_names)
                print (num_people)
                if num_people > 1:
                    print(f"More than 1 person detected in frame {frame_count}. Deleting original video.")
                    cap.release()
                    os.remove(input_video_path)
                    break
                frame_count += 1
            
                if frame_count >=3:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "D:/arslan/dataset/dfdc_train_part_3"
    process_videos_in_folder(folder_path)
