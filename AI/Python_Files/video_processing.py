'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''

import os
import cv2
import pickle
import time
import numpy as np
import subprocess
import psycopg2
from datetime import datetime
import pytz
import boto3
from botocore.exceptions import NoCredentialsError
import tempfile
import json
import threading
from facenet_pytorch import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model

'''
==== Real-Time Face Detection, Identification, and Data Collection ====

This part will detect faces within the camera's view and draw a box around each face
which will be used for the Facial Identification process.

Each face will be predicted by the machine learning model, if the person is in the
database, they given a label in the dislay window along with a blue border.

If the person is unknown, they will be given a red border and thier face cutout will
be saved to the S3 bucket

When a new machine learning model is uploaded to the S3 bucket, the system will
automatically reload the new model
'''

# Global Variables and S3 Configurations
s3 = boto3.client('s3')
bucket_name = '-'
folder_name = 'new_faces'
last_seen_times = {}
dict_lock = threading.Lock()
resource_lock = threading.Lock()
current_model = None
current_labels = None
new_labels_path = None
default_label = "locked_model"

# Database connection parameters
db_params = {
    "dbname": "-",
    "user": "-",
    "password": "-",
    "host": "-",
    "port": "-"
}

# Initialize MTCNN for face detection
detector = MTCNN(thresholds=[0.6, 0.8, 0.92])
image_width = 224
image_height = 224

# RTMP URLs
input_rtmp_url = "-"
output_rtmp_url = "-"

def load_model_from_path(model_path):
    """
    Loads a Keras model from a given file path.

    @param model_path: The file path where the Keras model is stored.
    @return: Loaded Keras model object.
    """
    with resource_lock:
        return load_model(model_path)

def load_labels_from_path(labels_path):
    """
    Loads labels from a pickle file.

    @param labels_path: The file path where the labels are stored in a pickle file.
    @return: A dictionary containing loaded labels.
    """
    with resource_lock:
        with open(labels_path, 'rb') as f:
            return {key: value for key, value in pickle.load(f).items()}

def download_file_from_s3(bucket_name, file_key, local_path):
    """
    Downloads a specified file from an S3 bucket to a local path.

    @param bucket_name: Name of the S3 bucket.
    @param file_key: Key of the file in the S3 bucket.
    @param local_path: The local file path where the file will be saved.
    @return: None
    """
    global s3
    try:
        s3.download_file(bucket_name, file_key, local_path)
        print(f"File {file_key} downloaded to {local_path}")
    except Exception as e:
        print(f"Error downloading {file_key} from S3: {e}")

def upload_face_to_s3(bucket_name, folder_name, image_data):
    """
    Uploads a face image to an S3 bucket, naming the file based on the current timestamp.

    @param bucket_name: Name of the S3 bucket.
    @param folder_name: Folder name in the S3 bucket where the file will be stored.
    @param image_data: Byte data of the image to be uploaded.
    @return: None
    """
    global s3
    timestamp = int(time.time())
    s3_file_name = f"{folder_name}/unknown_person_{timestamp}.jpg"
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(image_data)
        f.flush()
        try:
            s3.upload_file(f.name, bucket_name, s3_file_name)
        except NoCredentialsError:
            print("Credentials not available")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            os.unlink(f.name)

def reload_resource(file_path):
    """
    Reloads the machine learning model or labels from a given file path.

    @param file_path: The file path of the model or labels to be reloaded.
    @return: None
    """
    global current_model, current_labels, new_labels_path
    if file_path.endswith('cam3.keras'):
        time.sleep(10)
        current_model = load_model_from_path(file_path)
        current_labels = load_labels_from_path(new_labels_path)
    elif file_path.endswith('cam3.pickle'):
        new_labels_path = file_path

def listen_to_sqs():
    """
    Listens to an SQS queue for messages indicating that a new model or labels file has been uploaded to S3.

    @return: None
    """
    sqs = boto3.client('sqs')
    queue_url = '-'
    print("Listening to:", queue_url)

    while True:
        response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20)
        if 'Messages' in response:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']

            try:
                message_body = json.loads(message['Body'])
                file_key = message_body['file_key']

                if file_key.endswith('cam3.keras') or file_key.endswith('cam3.pickle'):
                    local_path = f'./{file_key}'
                    download_file_from_s3(bucket_name, file_key, local_path)
                    reload_resource(local_path)
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            except json.JSONDecodeError:
                print("Error processing message")
        time.sleep(1)

def try_predict_with_model(image):
    """
    Attempts to predict with the current model. If the model is locked for reloading, a default label is assigned.

    @param image: The image data to be predicted.
    @return: Tuple containing predicted probabilities, predicted name, and index of maximum prediction.
    """
    predicted_prob = None
    name = default_label
    maxIndex = None

    lock_acquired = resource_lock.acquire(timeout=0.01)
    if lock_acquired:
        try:
            predicted_prob = current_model(image)
            maxIndex = tf.argmax(predicted_prob, axis=1)[0].numpy()
            name = current_labels[maxIndex]
        finally:
            resource_lock.release()
    else:
        print("Model is being reloaded, assigning default label.")
    
    return predicted_prob, name, maxIndex

def update_database():
    """
    Periodically updates the database with the last seen times of identified people.

    @return: None
    """
    while True:
        time.sleep(120)
        with dict_lock:
            update_data = last_seen_times.copy()
            last_seen_times.clear()

        conn = None
        try:
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            for tag_name, timestamp in update_data.items():
                dt_utc = datetime.utcfromtimestamp(timestamp)
                central = pytz.timezone('-')
                dt_central = dt_utc.astimezone(central)
                formatted_time = dt_central.strftime("%I:%M%p")
                cur.execute("UPDATE example_table SET last_seen = %s WHERE tag_name = %s;", (formatted_time, tag_name))
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error in update operation: {error}")
        finally:
            if conn is not None:
                conn.close()

# Setup for face detection and streaming
stream = cv2.VideoCapture(input_rtmp_url)
if not stream.isOpened():
    print("Error: Couldn't open the RTMP stream.")
    exit()

stream_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
stream_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(stream.get(cv2.CAP_PROP_FPS))
process = subprocess.Popen([
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{stream_width}x{stream_height}",
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'flv',
    '-pix_fmt', 'yuv420p',
    '-b:v', '6000000',
    '-f', 'flv',
    output_rtmp_url
], stdin=subprocess.PIPE)

frame_count = 0
start_time = time.time()
display_fps = 0
identified_dict = {}

# Start background threads
sqs_listener_thread = threading.Thread(target=listen_to_sqs, daemon=True)
sqs_listener_thread.start()
db_update_thread = threading.Thread(target=update_database, daemon=True)
db_update_thread.start()

# start an infinite loop to capture feed until videos are over or user ends stream
while(True):
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()
    if frame is None:
        print("Failed to grab frame")
        continue
    
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
        if (w * h < 2500) :
           continue
        
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

        # predict the label with function
        predicted_prob, name, maxIndex = try_predict_with_model(img)

        # if model is locked
        if name == 'locked_model':
            # Draw a rectangle around the face
            color = (0, 0, 0) # Blue
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
        
        # if the face is recognized as a specific person
        elif name != '_Unknown':
            # add name to update dictionary for db update
            current_time = time.time()
            with dict_lock:
                last_seen_times[name] = current_time

            # Draw a rectangle around the face
            color = (255, 0, 0) # Blue
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
            color = (0, 0, 255) # Red
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

            if not(h >= 100 or w >= 100):
                continue

            # Encode the image to JPEG format
    	    # This step is necessary because you cannot directly upload a numpy array to S3
            _, buffer = cv2.imencode('.jpg', face_rgb)

   	    # Convert buffer to a byte string
            image_byte_data = buffer.tobytes()

    	    # Upload the face to S3
            upload_face_to_s3(bucket_name, 'new_faces', image_byte_data)

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

    # Calculate FPS every 10 seconds
    frame_count += 1
    if frame_count % (fps * 1) == 0:  # Calculate every 10 seconds
        elapsed_time = time.time() - start_time
        display_fps = frame_count / elapsed_time

    # Display FPS on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'FPS: {display_fps:.2f}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    #clear dictionary and release memory
    identified_dict.clear()
    
    # Write the frame to the output RTMP stream
    process.stdin.write(frame.tobytes())
    
# Cleanup
stream.release()
process.stdin.close()
process.wait()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

# Release memory
K.clear_session()