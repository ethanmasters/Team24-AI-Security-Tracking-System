'''
--------------------------------------------------------------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters

This is the code is to test the ML subsystem with preselected videos.
All processes will be explained within the code.

The base model was trained using a seperate file as it does not need to be continously ran.
This file is in the github as base_model_training.py
--------------------------------------------------------------------------------------------
The code used the following file strucure for datasets and buffer directories:
(Note: 'train' is the complete training datasets, 'inc_train' only contains 50 images of each pretrained class)
(Note: 'new_people_unsorted' and 'new_people_sorted' are buffers that will be populated by the code)
    
root/AI/
        new_people_unsorted/
        new_people_sorted/
        face_datasets/
            train/
                person1/
                    image1.png
                    image2.png
                person2/
                    image1.png
                    image2.png
            inc_train/
                person1/
                    image1.png
                    image2.png
                person2/
                    image1.png
                    image2.png
            valid/
                person1/
                    image1.png
                    image2.png
                person2/
                    image1.png
                    image2.png
            test/
                person1/
                    image1.png
                    image2.png
                person2/
                    image1.png
                    image2.png
        python_files/
            Models_and_Labels/
                model_1.h5
                model_1_labels.pickle
            Videos/
                train_video_1.mp4
                train_video_2.mp4
                train_video_3.mp4
                validation_video_1.mp4
                validation_video_2.mp4
            python_file_1.py
            python_file_2.py
            MachineLearningSubsystem.py
--------------------------------------------------------------------------------------------
'''

# standard imports
import os
import gc
import cv2
from PIL import Image
import pickle
import time
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt


# machine learning imports
import torch
from facenet_pytorch import MTCNN
from torchvision.models import vgg16
import torchvision.transforms as transforms

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input


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
previous_model = 'Models_and_Labels/base_face_cnn_model.h5'
model = load_model(previous_model)

# directory for saving new faces
directory = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_unsorted"

# open the labels for the trained model and save to a dictionary
previous_labels = 'Models_and_Labels/base-face-labels.pickle'
with open(previous_labels, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}

'''
This file is inteded to test using preselected videos. The videos were chosen becuase they
simulate an example use case environment of the final system, a pop-up shop/commerce checkout.

There are 3 videos that will be used to train, 2 videos used to validate.
All videos use the same 4 Actors, although only 3 will be present in the validation videos.
'''
# loop through each training video
for video in ["Videos/train_video_1.mp4", "Videos/train_video_2.mp4", "Videos/train_video_3.mp4"] :
    
    # start video capture of current video
    stream = cv2.VideoCapture(video)
    
    print(f"\nNOW WATCHING {video}\n")
    
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
        
        # check if the video is over and move to next video if so
        if not grabbed:
            print("End of video file")
            break
        
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
             
            '''
            REMOVED FOR THE TEST VIDEOS DUE TO THE SIZE OF THE VIDEO FILES
            WILL BE USED IN REAL-TIME IMPLEMENTATION
            #if face is too small (reasonably far from camera) we dont want to consider that person
            if (w * h < 2000) :
               continue
           '''
            
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
        cv2.imshow("Video", frame)
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

print("\nDONE WITH TRAINING VIDEOS, NOW SORTING STRANGERS\n")


'''
==== Sorting Stranger's Faces ====

We reach this process once the training videos are over.
In the final implementation, this code will trigger on a timer every 5 mintutes

A vgg16 model is used to calculate the embeddings (numerical representation of facial features) of the stranger's faces.
The "distance" or difference between the embeddings is used to determine if two faces belong to the same person.

Because the input datas are subdirectories of the same person, 10 random photos are chosen to compare the distance to
all the photos already sorted. After all 10 photos are done, the minimum distance of the 10 randomly chosen photos is used
to determine where the subdirectory gets sorted.

THIS PROCESS IS UNOPTIMIZED FOR RUNTIME AND WILL BE OMPTIMIZED TO RUN MUCH FASTER DURING THE SUMMER/403.
'''

# this was added to fix a bug
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# the distance value used to determine if faces are the same person or not
threshold = 0.25

# a list to keep all embeddings
embeddings = []

# if the system has a GPU, it will run on that, if not it will use CPU power
# GPU's are preferred for computer vision ML models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the VGG16 model
model = vgg16().to(device).eval()

# transform the input image to the expected size and format
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# input directory containing face images
input_dir = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_unsorted"

# loop through all subdirectories in the input directory and remove any that are less than 2 files long
# this filters any false positives that briefly get caught or faces that won't have enough data to train on
for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if os.path.isdir(subdir_path) and len(os.listdir(subdir_path)) <= 2:
        shutil.rmtree(subdir_path)

# output directory for sorted images
output_dir = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_sorted"

# get a list of all subdirectories in the input directory
person_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# loop over each person subdirectory
for person_dir in person_dirs:
    print(f"\nChecking {person_dir}")
    
    # get a list of all image files in the person directory
    image_files = [f for f in os.listdir(os.path.join(input_dir, person_dir)) if f.endswith('.jpg') or f.endswith('.png')]
    
    # randomly select up to 10 images from the person directory
    num_images = min(len(image_files), 10)
    images = np.random.choice(image_files, num_images, replace=False)
    
    # compute the embeddings of the selected images
    embeddings = []
    for filename in images:
        # load the image
        image_path = os.path.join(input_dir, person_dir, filename)
        image = Image.open(image_path)

        # apply the transformation and add a batch dimension
        img_tensor = transform(image).unsqueeze(0).to(device)

        # skip images where no face was detected
        if img_tensor is None:
            continue

        # compute the face embedding using the VGG16 model
        with torch.no_grad():
            features = model.features(img_tensor)
        embedding = features.view(-1).cpu().numpy()

        # normalize the embedding vector
        embedding /= np.linalg.norm(embedding)

        # add the embedding
        embeddings.append(embedding)

    # get a list of all subdirectories in the output directory
    person_paths = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    # initialize a list to keep track of the subdirectory the compared to photo is in
    photo_subdir = []
    
    # if this is the first directory to get checked
    if (len(person_paths) == 0):
        # Create a new person directory
        person_name = f"person{len(person_paths)}"
        person_path = os.path.join(output_dir, person_name)
        os.makedirs(person_path)
        print(f"Creating new person directory: {os.path.basename(person_path)}")

        # Copy the selected images to the person directory
        for filename in image_files:
            src_path = os.path.join(input_dir, person_dir, filename)
            dst_path = os.path.join(person_path, filename)
            shutil.copyfile(src_path, dst_path)

        # Add the new embeddings and person path to the lists
        person_paths.append(person_path)
        embeddings.extend(embeddings)
        
        # delete input subdirectory once sorted
        subdir_path = os.path.join(input_dir, person_dir)
        shutil.rmtree(subdir_path)
        
        continue
    
    # Compute the distances between the new embeddings and all previously seen embeddings
    if embeddings:
        # declare list to keep the distances
        distances = []
        
        #loop through all output subdirectories
        for i in range(len(person_paths)):
            person_path = person_paths[i]
            
            # get all photos of the subdirectory to iterate through
            person_images = [f for f in os.listdir(os.path.join(output_dir, person_path)) if f.endswith('.jpg') or f.endswith('.png')]
            for filename in person_images:
                # load the image
                image_path = os.path.join(output_dir, person_path, filename)
                image = Image.open(image_path)
                
                # add the subdirectory to the subdir list
                photo_subdir.append(person_path)

                # apply the transformation and add a batch dimension
                img_tensor = transform(image).unsqueeze(0).to(device)

                # skip images where no face was detected
                if img_tensor is None:
                    continue

                # compute the face embedding using the VGG16 model
                with torch.no_grad():
                    features = model.features(img_tensor)
                embedding = features.view(-1).cpu().numpy()

                # normalize the embedding vector
                embedding /= np.linalg.norm(embedding)

                # compute the distance between the new embedding and the current embedding
                distance = np.linalg.norm(embeddings - embedding, axis=1).mean()

                # add the distance to the distances list
                distances.append(distance)

        # Find the person with the smallest distance
        if distances:
            # get the index of the shortest ditance
            min_distance_idx = np.argmin(distances)
            
            # get the min distance value
            min_distance = distances[min_distance_idx]
            print(min_distance)
            
            # if the face is close enough to a already sorted person
            if min_distance < threshold:
                # get the output directory of the matched person
                person_path = photo_subdir[min_distance_idx]
                print(f"Adding images to existing person directory: {os.path.basename(person_path)}")
            else:
                # Create a new person directory
                person_name = f"person{len(person_paths)}"
                person_path = os.path.join(output_dir, person_name)
                os.makedirs(person_path)
                print(f"Creating new person directory: {os.path.basename(person_path)}")
        
            # Copy the selected images to the person directory
            for filename in image_files:
                src_path = os.path.join(input_dir, person_dir, filename)
                dst_path = os.path.join(output_dir, person_path, filename)
                shutil.copyfile(src_path, dst_path)
        
            # Add the new embeddings and person path to the lists
            person_paths.append(person_path)
            embeddings.extend(embeddings)
    
    # delete input subdirectory once sorted
    subdir_path = os.path.join(input_dir, person_dir)
    shutil.rmtree(subdir_path)

print("\nDONE WITH SORTING, NOW AUGMENTING AND SPLITTING DATA\n")


'''
==== Augmenting Data and Splitting Data into Training/Validaiton/Testing Sets ====

The image data generator a part of the incremental learning process later will do most of the
image augmentation. To assist this, half of the photos will be flipped horizontally before sent
to the model to be trained upon. This allows the model to predict people from a wider range of 
angles when facing the camera. 
(As the model can only predict with what it's trained with, if you were to only train the model 
 of a person facing the left, it would not be able to correctly predict them if they were 
 facing the right, this process eliminates that problem)

Once the images have been randomly flipped, they will be split from their sorted subdirectory into 3 seperate
data sets at different ratios. The split goes as follows:
        Training Set (80%): Used to train the model
        Validation Set (10%): Used to evaluate the model during the training process to optimize the learning
        Testing Set (10%): Used to test the model after training. Creating a seperate dataset ensures that the
                           data that is being tested has not been seen before.
                           (This data set will not be used during the real-time implementation of the subsystem
                            but is useful to check the functionality of the model during the development process)
'''

# set the paths for your input folder and output folders
input_folder = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_sorted"
output_folder = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/face_datasets"

# flip half the photos randomly
# loop through all subdirectories of root_dir
for subdir, dirs, files in os.walk(input_folder):
    # loop through all files in the subdirectory
    for file in files:
        # check if the file is an image
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # open the image using PIL
            img_path = os.path.join(subdir, file)
            img = Image.open(img_path)

            # randomly flip the image horizontally
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # save the modified image back to the original file
            img.save(img_path)

# split images into data sets
# set the ratio of training/validation/testing sets
training_ratio = 0.8
validation_ratio = 0.1
testing_ratio = 0.1

# get paths of dataset directories
complete_training_datasets_folder = os.path.join(output_folder, "train")
training_folder = os.path.join(output_folder, "inc_train")
validation_folder = os.path.join(output_folder, "valid")
testing_folder = os.path.join(output_folder, "test")

# get the list of subfolders in the input folder
person_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

# loop through each subdirectory
for person_folder in person_folders:
    # get the list of image filenames in the person folder
    image_filenames = [f for f in os.listdir(person_folder) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

    # shuffle the list of image filenames (to ensure an even split of collected data)
    random.shuffle(image_filenames)

    # split the image filenames into training, validation, and testing sets
    split_index1 = int(len(image_filenames) * training_ratio)
    split_index2 = int(len(image_filenames) * (training_ratio + validation_ratio))

    training_filenames = image_filenames[:split_index1]
    validation_filenames = image_filenames[split_index1:split_index2]
    testing_filenames = image_filenames[split_index2:]
    
    # get the name of the person from the folder path
    person_name = os.path.basename(person_folder)

    # create subfolders for the person in the train, valid, and test folders if they don't exist
    person_train_folder = os.path.join(training_folder, person_name)
    if not os.path.exists(person_train_folder):
        os.makedirs(person_train_folder)

    person_valid_folder = os.path.join(validation_folder, person_name)
    if not os.path.exists(person_valid_folder):
        os.makedirs(person_valid_folder)

    person_test_folder = os.path.join(testing_folder, person_name)
    if not os.path.exists(person_test_folder):
        os.makedirs(person_test_folder)

    # copy the training images to the training folder (both the transfer learning and the complete training sets directory)
    for filename in training_filenames:
        source_path = os.path.join(person_folder, filename)
        # transfer learning directory
        destination_path = os.path.join(training_folder, os.path.basename(person_folder), filename)
        shutil.copyfile(source_path, destination_path)
        # complete training sets directory
        destination_path = os.path.join(complete_training_datasets_folder, os.path.basename(person_folder), filename)
        shutil.copyfile(source_path, destination_path)

    # copy the validation images to the validation folder
    for filename in validation_filenames:
        source_path = os.path.join(person_folder, filename)
        destination_path = os.path.join(validation_folder, os.path.basename(person_folder), filename)
        shutil.copyfile(source_path, destination_path)

    # copy the testing images to the testing folder
    for filename in testing_filenames:
        source_path = os.path.join(person_folder, filename)
        destination_path = os.path.join(testing_folder, os.path.basename(person_folder), filename)
        shutil.copyfile(source_path, destination_path)
    
    # once split, remove it from the sorted buffer directory
    shutil.rmtree(person_folder)
    
print("\nDONE WITH FLIPPING AND SPLITTING, NOW UPDATING MODEL\n")


'''
==== Updating the ML Model Using Transfer Learning ====

The previous model can now be trained for the new classes. The previous model is loaded and the last classification
layer is replaced with a new one to account for the increase in the number of classes the model can predict.
During testing, it was found that the best results from only retraining the last 3 layers of the model.
Becuase the model saves the weights from the previous training, only a small number of previous learned faces
is trained upon to ensure the model does not lose accuracy on older classes. This is done to also greatly
reduce the runtime of the training process.

The training process uses the validation set to track the validation loss during the training process to know when to
stop training. When the validation loss stops decreasing and begins to increase, the model automatically stops training.
'''
    
# load the saved model
model = load_model(previous_model)

# load the existing class dictionary
with open(previous_labels, 'rb') as f:
    class_dictionary = pickle.load(f)

# set the last n layers as trainable for incremental training
for layer in model.layers[:-3]:
    layer.trainable = False
    
# use default data preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# import data from training set and ensure the images are in the correct size for model
train_set = train_datagen.flow_from_directory(
    'C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/face_datasets/inc_train',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# use default data preprocessing
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# import data from validation set and ensure the images are in the correct size for model
valid_set = valid_datagen.flow_from_directory(
    'C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/face_datasets/valid',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)   
    
# add a new output layer for the new classes
num_new_classes = len(train_set.class_indices.values())
new_output = Dense(num_new_classes, activation='softmax', name='New_layer')(model.layers[-2].output)
new_model = Model(inputs=model.input, outputs=new_output)

# recompile the model with the same optimizer and loss function
# the optimal learning rate was found during the development process
new_model.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# load the weights 
new_model.load_weights(previous_model, by_name=True)

# use early stopping to stop over training by tracking validation loss
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# train the model with new data
r = new_model.fit(train_set, batch_size=1, verbose=1, epochs=50, validation_data=valid_set, callbacks=[early_stopping])

# save the updated model
new_model_name = f"Models_and_Labels/{num_new_classes}_classes_face_cnn_model.h5"
new_model.save(new_model_name)

# update the previous model name
previous_model = new_model_name

# get the subfolder names from the train_set object
subfolder_names = train_set.class_indices.keys()
  
# assign an index to each new subfolder name that doesn't already exist in the class dictionary
index = len(class_dictionary)
for name in subfolder_names:
    if name not in class_dictionary.values():
        class_dictionary[index] = name
        index += 1

# these next two lines ensure the labels are in alphabetical order alighned with their index
# sort the dictionary items by their values
sorted_items = sorted(class_dictionary.items(), key=lambda x: x[1])

# create a new dictionary with the sorted items and reassign the keys
sorted_dict = {i: v for i, (_, v) in enumerate(sorted_items)}
    
# save the updated class dictionary back to the file
new_labels_name = f"Models_and_Labels/{num_new_classes}_classes-face-labels.pickle"
with open(new_labels_name, 'wb') as f:
    pickle.dump(sorted_dict, f)

# update the previous labels name
previous_labels = new_labels_name

# print the training validation loss plot to monitor training
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# release memory
K.clear_session()
gc.collect()

print("\nDONE WITH MODEL RETRAINING, NOW CLEANING TRAINING DATASET")


'''
==== Cleaning Training Directory For Next Training Session ====

Now that the new classes have already been trained on, they only need to be trained on a smaller amount
of data in future trainings. This process will randomly delete images in the transfer learning directory
so there are only 50 images of each person in their subdirectory.

The unknown person dataset will keep 1,000 images instead of 50 due to its large base training dataset
and to ensure that the accuracy of detecting new people as strangers is as high as possible without
over-infuencing the models predictions. This value was optimized during the development process.
'''

root_folder = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/face_datasets/inc_train"

for dirpath, dirnames, filenames in os.walk(root_folder):
    
    # skip the unknown dataset
    if '.Unknown' in dirpath:
        continue
        
    # Shuffle the list of files in the directory
    random.shuffle(filenames)
    
    # Determine how many files to keep (maximum of 50)
    num_to_keep = min(len(filenames), 50)
    
    # Delete the remaining files
    for file_name in filenames[num_to_keep:]:
        os.remove(os.path.join(dirpath, file_name))

print("\nDONE CLEANING TRAINING DATASET, NOW TESTING NEW MODEL ON VALIDATION VIDEOS\n")


'''
==== Validating Training From Preselected Videos ====

In the actual implementation, the code would load the new model once it is trained and continue
tracking people in the frame with the new model.
In this test version, the validation videos will be used to determine if the model can recognize
the same actors is was trained on. As mentioned before, one of the actors that appears in the training
videos is not in the validation videos. However, identifying 3 of the actors will be enough to
validate the accuracy of the new model.

The code here is exact same as the 'Real-Time Face Detection, Identification, and Data Collection' section.
'''

# using a MTCNN model to find the faces in the frame for face detection
# the thresholds were optimized to reduce false positives (mistaking objects as faces)
detector = MTCNN(thresholds=[0.6,0.8,0.92])

# size of the image to predict, the model is trained to take in 224x224 images
image_width = 224
image_height = 224

# load the pre-trained model
model = load_model(previous_model)

# directory for saving new faces
directory = "C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/new_people_unsorted"

# open the labels for the trained model and save to a dictionary
with open(previous_labels, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}

# loop through each validation video
for video in ["Videos/validation_video_1.mp4", "Videos/validation_video_2.mp4"] :
    
    print(f"\nNOW WATCHING {video}\n")
    
    # start video capture of current video
    stream = cv2.VideoCapture(video)
    
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
        
        # check if the video is over and move to next video if so
        if not grabbed:
            print("End of video file")
            break
        
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

print("\nDONE WITH VALIDATION VIDEOS\n")
print("\nThe End\n")