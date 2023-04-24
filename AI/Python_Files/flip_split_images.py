'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''


# standard imports
import os
from PIL import Image
import shutil
import random


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