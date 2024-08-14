import os
import shutil

# Define the two directories
src_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/Jayant/Arnab/Dataset'
dst_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27'

# Get a list of all folders in the source directory
src_folders = [f for f in os.listdir(
    src_dir) if os.path.isdir(os.path.join(src_dir, f))]

# Iterate over the folders in the source directory
for folder in src_folders:
    # Construct the full path to the folder in both directories
    src_folder_path = os.path.join(src_dir, folder)
    dst_folder_path = os.path.join(dst_dir, folder)

    # Check if the modified_ps.txt file exists in the destination folder
    modified_ps_path = os.path.join(dst_folder_path, 'modified_ps.txt')
    if not os.path.exists(modified_ps_path):
        # If it doesn't exist, copy it from the source folder
        src_modified_ps_path = os.path.join(src_folder_path, 'modified_ps.txt')
        shutil.copy2(src_modified_ps_path, dst_folder_path)
        print(
            f"Copied modified_ps.txt from {src_folder_path} to {dst_folder_path}")
