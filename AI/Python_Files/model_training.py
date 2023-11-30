'''
-----------------------------------------------------------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters

This is the code that builds and trains the ML model.
To train the model to recognize a stranger, multiple datasets were compiled
    to get a dataset of over 17,000 images that covers a very large range
    of demographics. 
    (The theory behind this technique is that an untrained person will more closely
     match one of the 17,000 people in this dataset than one of the pretrained people)
-----------------------------------------------------------------------------------------
'''

# standard imports
import gc
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import boto3
from botocore.exceptions import NoCredentialsError


# machine learning imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import VGG16


# function for saving to AWS S3 bucket
def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")

bucket_name = '-'  # Replace with your S3 bucket name

# set the keras backend to tensorflow
K = tf.keras.backend

# use default data preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# import data from training set and ensure the images are in the correct size for model
train_set = train_datagen.flow_from_directory(
    's3_bucket/face_datasets_test/train',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# use default data preprocessing
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# import data from validation set and ensure the images are in the correct size for model
valid_set = valid_datagen.flow_from_directory(
    's3_bucket/face_datasets_test/valid',
    target_size=(224,224),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# get the number of classes in the training set
NO_CLASSES = len(train_set.class_indices.values())

# build the model using the default vgg16 model
base_model = VGG16(include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3))

# add custom layers for own training images
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with size equal to number of classes
preds = Dense(NO_CLASSES, activation='softmax')(x)

# create a new model with the base model's original input and the new model's output
model = Model(inputs = base_model.input, outputs = preds)

# don't train the first 19 layers (these are apart of the default vgg16 model)
for layer in model.layers[:19]:
    layer.trainable = False
    
# set new layers as trainable (these are the classification layers)
for layer in model.layers[19:]:
    layer.trainable = True

# compile the model
# the optimal learning rate was found during the development process
model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# use early stopping to stop over training by tracking validation loss
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# train the model
r = model.fit(train_set, batch_size = 1, verbose = 1, epochs = 50, validation_data = valid_set, callbacks=[early_stopping])

# save the model locally and in S3
filename1 = 'Models_and_Labels/-cam1.keras'
filename2 = 'Models_and_Labels/-cam2.keras'
filename3 = 'Models_and_Labels/-cam3.keras'
model.save(filename1)
model.save(filename2)
model.save(filename3)

# create a mapping of index for each person in dataset
class_dictionary = train_set.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}

# save the dictionary
face_label_filename1 = 'Models_and_Labels/-cam1.pickle'
face_label_filename2 = 'Models_and_Labels/-cam2.pickle'
face_label_filename3 = 'Models_and_Labels/-cam3.pickle'
with open(face_label_filename1, 'wb') as f: pickle.dump(class_dictionary, f)
with open(face_label_filename2, 'wb') as f: pickle.dump(class_dictionary, f)
with open(face_label_filename3, 'wb') as f: pickle.dump(class_dictionary, f)
upload_to_s3(face_label_filename1, bucket_name, face_label_filename1)
upload_to_s3(face_label_filename2, bucket_name, face_label_filename2)
upload_to_s3(face_label_filename3, bucket_name, face_label_filename3)

time.sleep(10)
upload_to_s3(filename1, bucket_name, filename1)
upload_to_s3(filename2, bucket_name, filename2)
upload_to_s3(filename3, bucket_name, filename3)

# print the training validation loss plot
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("reports/-CURVES.png")
plt.clf()

# release memory
K.clear_session()
gc.collect()
