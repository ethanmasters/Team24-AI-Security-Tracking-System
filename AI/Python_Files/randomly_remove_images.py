'''
------------------------------------
Team: AI Security Tracking System
Author: Ethan Masters
------------------------------------
'''

# standard imports
import os
import random

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
