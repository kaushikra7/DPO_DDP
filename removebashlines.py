import os

# Define the two main folders
model = 'model18scnewonunscrambled'
folders = f'/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_scripts/{model}'
folderr = f'/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/{model}'
for filename in os.listdir(folders):
    if filename.endswith(".sh"):
        # Open the file in read mode
        with open(os.path.join(folders, filename), 'r') as file:
            # Read the lines of the file
            lines = file.readlines()

        # Iterate through each line
        new_lines = []
        for line in lines:
            # Check if the line contains the --output_file_path flag
            if '--output_file_path' in line:

                output_file_path_list = line.split(
                    '--output_file_path')[1].strip().split()

                output_file_path_list = output_file_path_list[:output_file_path_list.index(
                    '--parent_dir')]
                output_file_path = ''.join(output_file_path_list)
                output_file_path = output_file_path.replace('\\', ' ')
                # output_file_path = line.split('--output_file_path')[1].strip().split()
                output_file_path_txt = output_file_path[:-5] + '.txt'

                # Check if the output file exists
                # print(output_file_path_txt)
                if os.path.exists(output_file_path):
                    if os.path.exists(output_file_path_txt):
                        continue
                    os.remove(output_file_path)
                if os.path.exists(output_file_path_txt):
                    if os.path.exists(output_file_path):
                        continue
                    else:
                        os.remove(output_file_path_txt)

                        # If the line doesn't contain the --output_file_path flag or the file doesn't exist, add it to the new lines
            new_lines.append(line)
            # print(new_lines)
        print(new_lines)
        with open(os.path.join(folders, filename), 'w') as file:
            # with open(os.path.join(folders, 'temp.sh'), 'w') as file:
            for line in new_lines:
                # file.writelines(new_lines)
                file.write(line)
