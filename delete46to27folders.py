
import os
import shutil
# Define the two directories
# dir2 = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/datasetreasoning27'
# dir1 = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27'

# folders_in_dir1 = [f for f in os.listdir(
#     dir1) if os.path.isdir(os.path.join(dir1, f))]
# folders_in_dir2 = [f for f in os.listdir(
#     dir2) if os.path.isdir(os.path.join(dir2, f))]

# folders_to_delete = [f for f in folders_in_dir1 if f not in folders_in_dir2]

# for folder in folders_to_delete:
#     folder_path = os.path.join(dir1, folder)
#     print(f"Deleting folder: {folder_path}")
#     # shutil.rmtree(folder_path)
# print(len(folders_to_delete))
# print(folders_in_dir1)
# print(folders_in_dir2)

import os
import shutil

# src_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27'
# dst_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/datasetreasoning27'

# for folder in os.listdir(src_dir):
#     src_folder = os.path.join(src_dir, folder)
#     dst_folder = os.path.join(dst_dir, folder)
#     for file in ['ps.txt', 'modified_ps.txt', 'problemStatement.txt']:
#         src_file = os.path.join(src_folder, file)
#         dst_file = os.path.join(dst_folder, file)
#         if os.path.exists(src_file):
#             print(src_file)
#             shutil.copy2(src_file, dst_file)

src_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27'
dst_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/datasetreasoning27'

for folder in os.listdir(src_dir):
    src_folder = os.path.join(src_dir, folder)
    dst_folder = os.path.join(dst_dir, folder)
    src_file = os.path.join(src_folder, 'rubrics.json')
    dst_file = os.path.join(dst_folder, 'rubrics.json')
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
