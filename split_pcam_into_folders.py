import os
import shutil
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Paths to the folders and CSV file
images_folder = "/gris/gris-f/homestud/charder/deep-weight-prior/data/pcam/train"
csv_file = "/gris/gris-f/homestud/charder/deep-weight-prior/data/pcam/train_labels.csv"

# Paths to the output folders
output_folder = "/gris/gris-f/homestud/charder/deep-weight-prior/data/pcam_folders_full"
train_folder = os.path.join(output_folder, "train")
valid_folder = os.path.join(output_folder, "valid")
test_folder = os.path.join(output_folder, "test")

# Create the output folders and label subfolders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load the CSV file containing labels
labels_df = pd.read_csv(csv_file)
labels_df.set_index("id", inplace=True)

# Splitting ratio
train_ratio = 0.7
valid_ratio = 0.1
test_ratio = 0.2

# Shuffle the image filenames for each class separately
print("Getting labels")
labels_0 = labels_df[labels_df["label"] == 0].index.tolist()
labels_1 = labels_df[labels_df["label"] == 1].index.tolist()
random.shuffle(labels_0)
random.shuffle(labels_1)

# Determine the number of images for each split and each class
num_labels_0 = len(labels_0)
num_labels_1 = len(labels_1)

num_train_0 = int(train_ratio * num_labels_0)
num_valid_0 = int(valid_ratio * num_labels_0)
num_test_0 = num_labels_0 - num_train_0 - num_valid_0

num_train_1 = int(train_ratio * num_labels_1)
num_valid_1 = int(valid_ratio * num_labels_1)
num_test_1 = num_labels_1 - num_train_1 - num_valid_1

# Function to copy images to their respective folders
def copy_images_to_folders(filenames, folder_path, label):
    for filename in tqdm(filenames):
        image_path = os.path.join(images_folder, filename+".tif")

        label_folder = os.path.join(folder_path, str(label))
        os.makedirs(label_folder, exist_ok=True)
        shutil.copy(image_path, label_folder)

# Split the images and copy to the respective folders for each class
train_images_0 = labels_0[:num_train_0]
train_images_1 = labels_1[:num_train_1] 
valid_images_0 = labels_0[num_train_0:num_train_0 + num_valid_0] 
valid_images_1 = labels_1[num_train_1:num_train_1 + num_valid_1]
test_images_0 = labels_0[num_train_0 + num_valid_0:] 
test_images_1 =  labels_1[num_train_1 + num_valid_1:]

copy_images_to_folders(train_images_0, train_folder, 0)
copy_images_to_folders(train_images_1, train_folder, 1)
copy_images_to_folders(valid_images_0, valid_folder, 0)
copy_images_to_folders(valid_images_1, valid_folder, 1)
copy_images_to_folders(test_images_0, test_folder, 0)
copy_images_to_folders(test_images_1, test_folder, 1)

print("Splitting and organizing images completed successfully.")
