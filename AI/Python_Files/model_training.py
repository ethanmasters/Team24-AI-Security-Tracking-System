'''
-----------------------------------------------------------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters

This is the code that builds and trains the base ML model.
The model was trained on two people sudents: Ethan Masters and Adrian Duarte.
The model was also trained on a handful of celebrity photos.
To train the model to recognize a stranger, multiple datasets were compiled
    to get a dataset of over 17,000 images that covers a very large range
    of demographics. 
    (The theiry behind this technique is that an untrained person will more closely
     match one of the 17,000 people in this dataset than one of the pretrained people)
-----------------------------------------------------------------------------------------
'''

# standard imports
import gc
import pickle
import matplotlib.pyplot as plt


# machine learning imports
from keras import backend as K
from tensorflow.keras.models import Model
from keras_vggface.vggface import VGGFace
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input


# use default data preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# import data from training set and ensure the images are in the correct size for model
train_set = train_datagen.flow_from_directory(
    'C:/Users/ethan/Desktop/Capstone/Team24-AI-Security-Tracking-System/AI/face_datasets/train',
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

# get the number of classes in the training set
NO_CLASSES = len(train_set.class_indices.values())

# build the model using the default vgg16 model
base_model = VGGFace(include_top=False,
    model='vgg16',
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

# save the model
model.save('Models_and_Labels/base_face_cnn_model.h5')

# create a mapping of index for each person in dataset
class_dictionary = train_set.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}

# save the dictionary
face_label_filename = 'Models_and_Labels/base-face-labels.pickle' #for real-time use 'realtime-face-labels.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)

# print the training validation loss plot
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