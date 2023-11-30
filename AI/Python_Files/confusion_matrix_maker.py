'''
----------------------------------------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters

This code is used to generate the confusion matrix of a saved model.
This uses the testing dataset to test the model on newly seen images.
The code also generates the statistics of the confusion matrix for
    every class and the total statistics
----------------------------------------------------------------------
'''

# standard imports
import os
import pickle
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

# machine learning imports
from tensorflow import keras
from tensorflow.keras.models import load_model


# load the pre-trained model
model = load_model('Models_and_Labels/-cam1.keras')

# define the path to the testing data
testing_dir = "s3_bucket/face_datasets_test/test"

# create a list of all the subdirectories in the testing directory
subdirs = os.listdir(testing_dir)

# load the existing class dictionary
with open('Models_and_Labels/-cam1.pickle', 'rb') as f:
    class_dict = pickle.load(f)
    
# swap keys and values in class_dict
class_dict_new = {v:k for k,v in class_dict.items()}
    
# create empty lists to store the predicted and actual classes
predicted_classes = []
actual_classes = []

# loop through each subdirectory in the testing directory
for subdir in subdirs:
    # get a list of all the image files in the subdirectory
    image_files = os.listdir(os.path.join(testing_dir, subdir))
    # loop through each image file in the subdirectory
    for image_file in image_files:
        # load the image file as an array
        image = keras.preprocessing.image.load_img(os.path.join(testing_dir, subdir, image_file), target_size=(224, 224))
        image_array = keras.preprocessing.image.img_to_array(image)
        # make a prediction for the image using the trained model
        prediction = model.predict(np.array([image_array]))[0]
        # append the predicted class and actual class to their respective lists
        predicted_classes.append(np.argmax(prediction))
        actual_classes.append(class_dict_new[subdir])

# use scikit-learn to create the confusion matrix
cm = confusion_matrix(actual_classes, predicted_classes)

# generate the statistics report
report = classification_report(actual_classes, predicted_classes)

# Save the classification report to a text file
report_dir = "reports"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

report_file_path = os.path.join(report_dir, "-classification_report.txt")
with open(report_file_path, 'w') as report_file:
    report_file.write(report)

# Plot the confusion matrix and save the plot as an image
fig, ax = plot_confusion_matrix(
    conf_mat=cm,
    class_names=list(class_dict_new.keys()),
    colorbar=True,
    show_normed=True,
    show_absolute=False,
    figsize=(20, 20)
)

confusion_matrix_file_path = os.path.join(report_dir, "-confusion_matrix.png")
fig.savefig(confusion_matrix_file_path)
