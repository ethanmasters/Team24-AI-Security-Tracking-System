'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''

# standard imports
import os
import cv2
import pickle
import time
import numpy as np


# machine learning imports
from facenet_pytorch import MTCNN

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model


'''
==== Real-Time Face Detection, Identification, and Data Collection ====

This part will detect faces within the camera's view and draw a box around each face
which will be used for the Facial Identification process.

Each face will be predicted by the machine learning model, if the person is in the
database, they given a label in the dislay window along with a blue border.

If the person is unknown, they will be given a red border and thier face cutout will
be saved to a buffer directory. The cutouts will be saved in their own subdirectory
for as long as the person is in view of the camera.
'''

# using a MTCNN model to find the faces in the frame for face detection
# the thresholds were optimized to reduce false positives (mistaking objects as faces)
detector = MTCNN(thresholds=[0.6,0.8,0.92])

# size of the image to predict, the model is trained to take in 224x224 images
image_width = 224
image_height = 224

# load the pre-trained model
previous_model = 'Models_and_Labels/20_classes_face_cnn_model.h5'
model = load_model(previous_model)

# directory for saving new faces
directory = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_unsorted"

# open the labels for the trained model and save to a dictionary
previous_labels = 'Models_and_Labels/20_classes-face-labels.pickle'
with open(previous_labels, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
  
# start video capture of webcam
stream = cv2.VideoCapture(0)

# set the display window output dimensions
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# initialize the FPS counter and timer
fps = 0
fps_counter = 0
fps_start_time = 0

# initialize dictionary to store identified faces
identified_dict = {}
# initialize dictionary to store unknown faces
stranger_dict = {}

# initialize dictionary to store centerpoints from previous frames
frame_buffer = {}
# initialize value to keep track of frames that has passed (used to clear frame buffer every n frames)
num_frames = 0

# start an infinite loop to capture feed until videos are over or user ends stream
while(True):
    # increment our frame counter
    num_frames += 1
    
    # clear frame buffer every 10 frames
    '''
    the purpose of this is so that if the camera loses someones face for only a brief period of
    time, it will still remember that there was a person in that location. This cleans up how
    many unknown subdirectories are created by the Data Collection Process.
    '''
    if (num_frames % 10 == 0):
        frame_buffer.clear()
    
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()
    
    # check if the frame is not empty
    if frame is not None:
        # convert the color of the frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    # landmarks=False because we only care about the bounding box, not the location of the eyes, nose, or mouth
    faces, _ = detector.detect(frame, landmarks=False)
    
    # if one or more faces is found
    if faces is not None:
        # save the bounding box paramaters in a list
        bounding_boxes = faces[:, :4]
    else:
        #set the bounding box list to empty
        bounding_boxes = []

    # for each faces found
    for face in bounding_boxes:
        # get the values for the (x, y) coordinate pair of the upper-left corner of the box 
        x = int(face[0])
        y = int(face[1])
        
        # get the values of the width and height of the box
        w = int(face[2] - face[0])
        h = int(face[3] - face[1])
        
        # if the box's origin is not in frame, set the origin to be the edge of the frame and adjust the width/height
        # if to the left of the frame
        if x < 1:
            w = w + x - 1
            x = 1
        # if above the frame
        if y < 1:
            h = h + y - 1
            y = 1
           
        #if face is too small (reasonably far from camera) we dont want to consider that person
        if (w * h < 2000) :
           continue
        
        #get center dot to use to track distance between frames
        centerX = int(x + (w / 2))
        centerY = int(y + (h / 2))
        
        # get the bounding box as an array of pixels
        face_rgb_buff = rgb[y:y+h, x:x+w]
        # save color as RGB
        face_rgb = cv2.cvtColor(face_rgb_buff, cv2.COLOR_BGR2RGB)
        
        # resize the image to work for the ML model
        size = (image_width, image_height)
        resized_image = cv2.resize(face_rgb, size)
        
        # convert image to numpy array, uint8 is the datatype (unsigned integer with 8 bits)
        image_array = np.array(resized_image, "uint8")
        
        # reshape to 4D tensor for ML model. 1 image, 3 color channels
        img = image_array.reshape(1,image_width,image_height,3) 
        
        # cast as single precision data type
        img = img.astype('float32')
        
        # convert the pixels values from [0, 255] -> [0, 1]
        img /= 255

        # predict the image
        predicted_prob = model(img)
        
        # free memory from prediction
        K.clear_session()
        
        # get the index of the person with the highest confidence of prediciton
        maxIndex = tf.argmax(predicted_prob, axis=1)[0].numpy()
        
        # get the label of the prediction with the highest confidence
        name = labels[maxIndex]
        
        # if the face is recognized as a specific person
        if name != '.Unknown':  
            # Draw a rectangle around the face
            color = (255, 0, 0) # in BGR
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            
            # save the name of the person and the confidence value in a dictionary with the respective bounding box
            # only save most confident box for each person
            # if two faces are predicted as the same person, the one with the lowest confidence will get a blue box with no label
            if name not in identified_dict or predicted_prob[0][maxIndex] > identified_dict[name]['prob']:
                identified_dict[name] = {'prob': predicted_prob[0][maxIndex], 'box': (x, y, w, h)}
        
        # logic for data collection for unknown faces
        else:
            # Draw a rectangle around the face
            color = (0, 0, 255) # in BGR
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
            
            # get the current time to use as a file name/photo label
            currTime = time.time()
            
            # if this face is the first stranger detected
            if len(stranger_dict) == 0:
                # if it is, add the face to a new subdirectory
                new_dir_name = f"person_{currTime}"
                
                # position to stranger dictionary and frame buffer
                stranger_dict[new_dir_name] = (centerX, centerY)
                frame_buffer[new_dir_name] = (centerX, centerY)
                
                # make a new subdirectory for the new person
                os.makedirs(os.path.join(directory, new_dir_name))
                
                # save the face to the subdirectoy
                cv2.imwrite(os.path.join(directory, new_dir_name, f"unknown_person_{currTime}.jpg"), face_rgb)
            
            # case when the face is not the first stranger detected
            else:
                # loop over the stranger_dict to check if the face is already in it
                for key, value in stranger_dict.items():
                    # calculate the distance between the current face and the center of the bounding box in the stranger_dict
                    distance = ((centerX - value[0])**2 + (centerY - value[1])**2)**0.5
                    # if the distance is smaller than a threshold, add the face to the current subdirectory
                    if distance < 50:
                        stranger_dict[key] = (centerX, centerY)
                        frame_buffer[key] = (centerX, centerY)
                        cv2.imwrite(os.path.join(directory, key, f"unknown_person_{currTime}.jpg"), face_rgb)
                        break
                    
                else:
                    # if the face is not already in the stranger_dict or no subdirectory has a close enough bounding box center,
                    # add the face to a new subdirectory
                    new_dir_name = f"person_{currTime}"
                    stranger_dict[new_dir_name] = (centerX, centerY)
                    frame_buffer[new_dir_name] = (centerX, centerY)
                    os.makedirs(os.path.join(directory, new_dir_name))
                    cv2.imwrite(os.path.join(directory, new_dir_name, f"unknown_person_{currTime}.jpg"), face_rgb)
    
    # set the stranger dictionary as a copy of the frame buffer so the buffer can be cleared with no issue
    stranger_dict = frame_buffer.copy()

    # logic for adding labels to identified persons
    for name in identified_dict:
        prob_str = '{:.2f}%'.format(identified_dict[name]['prob'] * 100)
        x, y, w, h = identified_dict[name]['box']
        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 255)
        stroke = 1
        
        # default add label to top of box unless the box is at the top of the screen
        if (y - 8 <= 0):
            cv2.putText(frame, f'({name}: {prob_str})', (x,y+h+20), font, 0.75, color, stroke, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'({name}: {prob_str})', (x,y-8), font, 0.75, color, stroke, cv2.LINE_AA)

    # update FPS counter every two frames 
    fps_interval = 2
    fps_counter += 1
    if fps_counter == 1:
        fps_start_time = time.time()
    elif fps_counter == fps_interval:
        fps = fps_interval / (time.time() - fps_start_time)
        fps_counter = 0

    # draw FPS on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    stroke = 2
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), font, 1, color, stroke, cv2.LINE_AA)
    
    #clear dictionary and release memory
    identified_dict.clear()
    K.clear_session()
    
    # Show the frame
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out of the loop
        print("Closed by User")
        break

# Cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

#release memory
K.clear_session()