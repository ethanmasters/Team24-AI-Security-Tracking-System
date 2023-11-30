'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''

# Standard imports
import os
import shutil
import random
from PIL import Image

# Additional imports for augmentation
import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, RandomGamma, GaussianBlur
from albumentations import ShiftScaleRotate, GaussNoise, HueSaturationValue

'''
==== Augmenting Data and Splitting Data into Training/Validaiton/Testing Sets ====

The image data generator a part of the incremental learning process later will do additional
image augmentation. To assist this, all of the photos will be copied and flipped horizontally 
and then manually augmented before sent to the model to be trained upon. This allows the model
to predict people from a wider range of angles when facing the camera and also. 
(As the model can only predict with what it's trained with, if you were to only train the model 
 of a person facing the left, it would not be able to correctly predict them if they were 
 facing the right, this process eliminates that problem)
After doubling the amount of images by flipping, random augmetation will be ran on the images
to generate even more data for the model training process and an additional augmentedcopy will
be saved, generating 4 output images for each input image:
    - Original Image
    - Flipped Image
    - Original Augmented
    - Flipped Augmented

Once the images have been flipped and augmetned, they will be split from their sorted subdirectory
into 3 seperate data sets at different ratios. The split goes as follows:
        Training Set (80%): Used to train the model
        Validation Set (10%): Used to evaluate the model during the training process to optimize the learning
        Testing Set (10%): Used to test the model after training. Creating a seperate dataset ensures that the
                           data that is being tested has not been seen before.
                           (This data set will not be used during the real-time implementation of the subsystem
                           but is useful to check the functionality of the model during the development process)
'''

# set the paths for your input folder and output folders
input_folder = "s3_bucket/test"
output_folder = "s3_bucket/face_datasets_test"

# Function to apply augmentations
def apply_augmentations(image_path, save_dir, file_prefix, img_size=(224,224)):
    """
    Applies a series of augmentations to an image and saves the result.

    @param image_path: Path to the original image.
    @param save_dir: Directory where the augmented image will be saved.
    @param file_prefix: Prefix to be added to the filename of the saved image.
    @param img_size: Size to which the image will be resized (default is 224x224).
    @return: None
    """
    # Define the augmentation pipeline
    augmentations = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, border_mode=0, p=0.5),
        GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ])
    
    # Read and augment the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = augmentations(image=image)
    augmented_image = augmented['image']
    
    # Save the augmented image
    filename, file_extension = os.path.splitext(os.path.basename(image_path))
    new_filename = "{}_{}{}".format(file_prefix, filename, file_extension)
    cv2.imwrite(os.path.join(save_dir, new_filename), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# Create flipped images
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(subdir, file)
            img = Image.open(img_path)
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            file_root, file_ext = os.path.splitext(file)
            flipped_img_path = os.path.join(subdir, "{}_flipped{}".format(file_root, file_ext))
            flipped_img.save(flipped_img_path)
            apply_augmentations(img_path, subdir, 'aug')
            apply_augmentations(flipped_img_path, subdir, 'aug_flipped')

# Set the ratio of training/validation/testing sets
training_ratio = 0.8
validation_ratio = 0.1
testing_ratio = 0.1

# Get paths of dataset directories
complete_training_datasets_folder = os.path.join(output_folder, "train")
training_folder = os.path.join(output_folder, "inc_train")
validation_folder = os.path.join(output_folder, "valid")
testing_folder = os.path.join(output_folder, "test")

# Loop through each person's folder in the input folder
person_folders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

for person_folder in person_folders:
    image_filenames = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_filenames)
    split_index1 = int(len(image_filenames) * training_ratio)
    split_index2 = int(len(image_filenames) * (training_ratio + validation_ratio))

    training_filenames = image_filenames[:split_index1]
    validation_filenames = image_filenames[split_index1:split_index2]
    testing_filenames = image_filenames[split_index2:]

    person_name = os.path.basename(person_folder)

    # Create subfolders for training, validation, and testing
    person_complete_training_datasets_folder = os.path.join(complete_training_datasets_folder, person_name)
    person_train_folder = os.path.join(training_folder, person_name)
    person_valid_folder = os.path.join(validation_folder, person_name)
    person_test_folder = os.path.join(testing_folder, person_name)
    os.makedirs(person_complete_training_datasets_folder, exist_ok=True)
    os.makedirs(person_train_folder, exist_ok=True)
    os.makedirs(person_valid_folder, exist_ok=True)
    os.makedirs(person_test_folder, exist_ok=True)

    # Copy images to respective folders
    for filename in training_filenames:
        source_path = os.path.join(person_folder, filename)
        destination_path = os.path.join(training_folder, person_name, filename)
        shutil.copyfile(source_path, destination_path)
        destination_path = os.path.join(complete_training_datasets_folder, person_name, filename)
        shutil.copyfile(source_path, destination_path)

    for filename in validation_filenames:
        source_path = os.path.join(person_folder, filename)
        destination_path = os.path.join(validation_folder, person_name, filename)
        shutil.copyfile(source_path, destination_path)

    for filename in testing_filenames:
        source_path = os.path.join(person_folder, filename)
        destination_path = os.path.join(testing_folder, person_name, filename)
        shutil.copyfile(source_path, destination_path)

    shutil.rmtree(person_folder)