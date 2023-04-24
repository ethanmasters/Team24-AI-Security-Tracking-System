'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''

# standard imports
import gc
import pickle
import matplotlib.pyplot as plt

# machine learning imports
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input


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

#define the previous model and labels
previous_model = 'Models_and_Labels/base_face_cnn_model.h5'
previous_labels = 'Models_and_Labels/base-face-labels.pickle'
    
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