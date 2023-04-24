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
import numpy as np


# machine learning imports
import torch
from torchvision.models import vgg16
import torchvision.transforms as transforms


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