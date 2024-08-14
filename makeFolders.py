import os
import shutil

src_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset3'
dst_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/few_shot_results'
# dst_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model_1'

# Create the destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Walk through the source directory and copy the folder structure
# for root, dirs, files in os.walk(src_dir):
#     for dir in dirs:
#         dir_path = os.path.join(root, dir)
#         print(dir)
#         rel_path = os.path.relpath(dir_path, src_dir)
#         dst_path = os.path.join(dst_dir, dir)
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)

list = []
for dir in os.listdir(src_dir):
    dir_path = os.path.join(dst_dir, dir)
    list.append(dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


print(list)
