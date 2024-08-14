import os

# Define the two main folders
folder1 = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18r'
# folder2 = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/dpo_results/dpo_model_2_final'
folder2 = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18scnewonsc'

# Get the list of subfolders in each main folder
subfolders1 = [f.name for f in os.scandir(folder1) if f.is_dir()]
subfolders2 = [f.name for f in os.scandir(folder2) if f.is_dir()]

# Get the common subfolders
common_subfolders = set(subfolders1) & set(subfolders2)

# Create a list to store the results
results = []
diff = 0
total = 0
# Iterate over the common subfolders
for subfolder in common_subfolders:
    # Get the number of files in the subfolder in each main folder
    num_files1 = len([f for f in os.scandir(
        os.path.join(folder1, subfolder)) if f.is_file()])
    num_files2 = len([f for f in os.scandir(
        os.path.join(folder2, subfolder)) if f.is_file()])

    # Append the results to the list
    results.append([subfolder, num_files1, num_files2])
    diff += num_files1-num_files2
    total += num_files1

# Print the results
for result in results:
    print(result)
print(diff)
print(total)


for subfolder in common_subfolders:
    # Get the list of files in the subfolder in each main folder
    files1 = [f.name for f in os.scandir(
        os.path.join(folder1, subfolder)) if f.is_file()]
    files2 = [f.name for f in os.scandir(
        os.path.join(folder2, subfolder)) if f.is_file()]

    # Find the files that are present in folder1 but not in folder2
    files_only_in_folder1 = [f for f in files1 if f not in files2]
    # Find the files that are present in folder2 but not in folder1
    files_only_in_folder2 = [f for f in files2 if f not in files1]

    # Print the files that are present in one folder but not the other
    if files_only_in_folder1:
        print(f"Files only in {folder1}/{subfolder}: {files_only_in_folder1}")
    if files_only_in_folder2:
        print(f"Files only in {folder2}/{subfolder}: {files_only_in_folder2}")

    # Get the number of files in the subfolder in each main folder
    num_files1 = len(files1)
    num_files2 = len(files2)

    # Append the results to the list
    results.append([subfolder, num_files1, num_files2])
    diff += num_files1 - num_files2
    total += num_files1
