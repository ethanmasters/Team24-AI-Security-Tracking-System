import os
import numpy as np
from PIL import Image
from torchvision.models import vgg16
from torchvision.transforms import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import shutil
import torch
import datetime
import pytz
import psycopg2
import boto3
from botocore.exceptions import NoCredentialsError

# Database connection parameters
db_params = {
    "dbname": "-",
    "user": "-",
    "password": "-",
    "host": "-",
    "port": "-"
}

# S3 bucket details
bucket_name = '-'  # Replace with your bucket name
s3_folder = 'new_faces/'  # Replace with your specific folder name in S3
local_dir = './s3_bucket/new_faces'  # Replace with the path to your local directory

# Load VGG16 model
model = vgg16(pretrained=True).eval()
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Input and output directories
input_dir = "s3_bucket/new_faces"
output_dir = "s3_bucket/new_faces_sorted"
complete_dataset_dir = "s3_bucket/CompleteDatasets"

# Function for downloading local copy of S3 bucket content
def download_directory_from_s3(bucket_name, s3_folder, local_dir):
    """
    Downloads an entire directory from an S3 bucket to a local directory.

    @param bucket_name: Name of the S3 bucket.
    @param s3_folder: Folder path in the S3 bucket.
    @param local_dir: Local directory to which the folder will be downloaded.
    @return: None
    """
    print("Now downloading from S3")
    s3 = boto3.client('s3')
    try:
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

        for page in page_iterator:
            if "Contents" in page:
                for key in page["Contents"]:
                    rel_path = key["Key"][len(s3_folder):]  # Relative path
                    local_file_path = os.path.join(local_dir, rel_path)

                    if not os.path.exists(local_file_path):
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        s3.download_file(bucket_name, key["Key"], local_file_path)
                    else:
                        print(f"File already exists: {local_file_path}")
        print("Download completed!")
    except NoCredentialsError:
        print("Credentials not available")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function for clearing the S3 folder after download
def delete_all_photos_from_s3(bucket_name, s3_folder):
    """
    Deletes all photos in a specified folder of an S3 bucket.

    @param bucket_name: Name of the S3 bucket.
    @param s3_folder: Folder path in the S3 bucket.
    @return: None
    """
    s3 = boto3.client('s3')
    try:
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)
        keys_to_delete = []

        for page in page_iterator:
            if "Contents" in page:
                for key in page["Contents"]:
                    keys_to_delete.append({'Key': key['Key']})

        if keys_to_delete:
            print("Deleting files from S3 bucket...")
            delete_response = s3.delete_objects(Bucket=bucket_name, Delete={'Objects': keys_to_delete})
            print("Files deleted from S3 bucket:", delete_response)
        else:
            print("No files to delete in S3 bucket.")
    except NoCredentialsError:
        print("Credentials not available")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function for adding a new person to the database
def add_person_to_table(file_name, db_params):
    """
    Adds a new person's data to the database using the specified file name.

    @param file_name: Name of the file, which includes a timestamp used for the database record.
    @param db_params: Database connection parameters.
    @return: None
    """
    timestamp_str = file_name.split("_")[-1].split(".")[0]
    timestamp = int(timestamp_str)
    dt_utc = datetime.utcfromtimestamp(timestamp)
    central = pytz.timezone('US/Central')
    dt_central = dt_utc.astimezone(central)
    formatted_time = dt_central.strftime("%I:%M%p")
    tag_name = file_name.rsplit('.', 1)[0]

    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO example_table (tag_name, first_seen, last_seen)
            VALUES (%s, %s, %s);
        """, (tag_name, formatted_time, formatted_time))
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error in add operation: {error}")
    finally:
        if conn is not None:
            conn.close()

# Process for downloading and organizing images
download_directory_from_s3(bucket_name, s3_folder, local_dir)
delete_all_photos_from_s3(bucket_name, s3_folder)

data_dict = {'embeddings': [], 'image_paths': []}
print(f"Processing images in {input_dir}")
for image_file in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_file)
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        aligned_image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(aligned_image_tensor).flatten().numpy()
            data_dict['embeddings'].append(embedding)
            data_dict['image_paths'].append(image_path)

embeddings_array = np.array(data_dict['embeddings'])
num_embeddings = len(embeddings_array)
if num_embeddings < 2:
    print("Not enough embeddings for clustering. Exiting.")
    exit()

min_clusters = 2
max_clusters = min(11, num_embeddings)
silhouette_scores = []
for num_clusters in range(min_clusters, max_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    silhouette_avg = silhouette_score(embeddings_array, cluster_labels)
    silhouette_scores.append(silhouette_avg)

if silhouette_scores:
    optimal_clusters = np.argmax(silhouette_scores) + min_clusters
    print(f"\nEstimated Number of People: {optimal_clusters}")
else:
    print("Could not estimate the number of clusters due to insufficient data.")
    exit()

kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(embeddings_array)
clusters_dict = {cluster_label: [] for cluster_label in range(optimal_clusters)}
for i, cluster_label in enumerate(cluster_labels):
    clusters_dict[cluster_label].append(data_dict['image_paths'][i])

# Create output folders, copy images, and save person data to the database
for cluster_label, image_paths in clusters_dict.items():
    if image_paths:
        first_image_path = image_paths[0]
        file_name = os.path.basename(first_image_path)
        person_folder_name = f"Person_{cluster_label}"
        person_folder = os.path.join(output_dir, person_folder_name)
        os.makedirs(person_folder, exist_ok=True)
        add_person_to_table(file_name, db_params)

        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(person_folder, image_filename))

        complete_dataset_folder = os.path.join(complete_dataset_dir, person_folder_name)
        shutil.copytree(person_folder, complete_dataset_folder, dirs_exist_ok=True)

        print(f"Processed and added Person_{cluster_label} to the database and complete dataset directory.")