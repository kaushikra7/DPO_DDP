from dataset_utils import get_rubrics
import os
import sys
sys.path.append('..')


def make_folders(parent_dir, output_dir):
    '''
    Creates files with the same names in output_dir as the ones in parent_dir
    '''
    for lab in os.listdir(parent_dir):
        os.mkdir(os.path.join(output_dir, lab))


if __name__ == "__main__":
    parent_dir = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27-scramblenew"
    results_dir = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18scnewonunscrambled"
    make_folders(parent_dir, results_dir)

    devices = ["cuda:1", "cuda:1", "cuda:1", "cuda:1"]
    eval_labs = [
        "CP_00104_loop_cs101f22_LE02_B_Q1",
        "CP_00105_sort_cs101f22_LE2_E_Q2",
        "CP_00107_sort_cs101f22_LE03_B_Q1",
        "CP_00111_array_cs101f22_LE02_B_Q2",
        "CP_00115_sort_cs101f22_LE03_C_Q1",
        "cs101a23_lq01_d_q4",
        "CP_00211_condition_cs101s23_LQ01_C_Q3",
        "CP_00108_loop_cs101f22_LE01_B_Q2",
        "CP_00214_var_cs101s23_lq01_d_q2"
    ]
    # eval_labs = [
    #     'cs101a23_lq01_a_q3',
    #     'CP_00214_var_cs101s23_lq01_d_q2',
    #     'cs101a23_lq01_a_q2',
    #     'CP_00108_loop_cs101f22_LE01_B_Q2',
    #     'CP_00211_condition_cs101s23_LQ01_C_Q3',
    #     'cs101a23_lq01_d_q3',
    #     'CP_00206_var_cs101s23_lq01_b_q2'
    # ]
    # eval_labs = [
    #     "CP_00102_loop_cs101f22_LE01_A_Q2",
    #     "CP_00214_var_cs101s23_lq01_d_q2",
    #     "cs101a23_lq01_b_q4",
    #     "CP_00106_loop_cs101f22_LE2_A_Q2",
    #     "cs101a23_lq01_a_q3",
    #     "cs101a23_lq01_a_q2",
    #     "CP_00108_loop_cs101f22_LE01_B_Q2",
    #     "CP_00211_condition_cs101s23_LQ01_C_Q3",
    #     "cs101a23_lq01_d_q3"
    # ]
    # eval_labs = ['CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1', 'CP_00315_var_cs101a23_lq01_c_q5',
    #  'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00216_var_cs101s23_LQ01_D_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00208_var_LQ01_B_Q4']
    # eval_labs = ["CP_00104_loop_cs101f22_LE02_B_Q1", "CP_00107_sort_cs101f22_LE03_B_Q1", "CP_00110_loop_cs101f22_LE01_C_Q2", "CP_00112_fun_cs101f22_LE02_C_Q1",
    #              "CP_00116_sort_cs101f22_LE03_E_Q1", "CP_00117_mat_cs101f22_LE03_B_Q2", "CP_00211_condition_cs101s23_LQ01_C_Q3", "CP_00315_var_cs101a23_lq01_c_q5"]
    # eval_labs = ["cs101a23_lq01_a_q3", "CP_00105_sort_cs101f22_LE2_E_Q2", "cs101s23_lq01_c_q4",
    #              "CP_00111_array_cs101f22_LE02_B_Q2", "CP_00108_loop_cs101f22_LE01_B_Q2"]
    # eval_labs = ["CP_00101_loop_cs101f22_LE01_A_Q1", "CP_00106_loop_cs101f22_LE2_A_Q2",
    #              "CP_00115_sort_cs101f22_LE03_C_Q1", "CP_00214_var_cs101s23_lq01_d_q2", "cs101a23_lq01_d_q4"]

    c1 = 0
    c2 = 0
    file_no = 2
    output_path = None
    num_files = 12

    first_time = [0]*num_files
    for lab in os.listdir(parent_dir):
        lab_path = os.path.join(parent_dir, lab)
        if (os.path.isdir(lab_path)):
            rubrics = get_rubrics(lab_path)
            for key in sorted(rubrics.keys()):
                device = devices[c2]
                tokens = key.split()
                output_file_name = "\ ".join(tokens)
                output_path = os.path.join(results_dir, lab)
                output_path = os.path.join(
                    output_path, f"{output_file_name}.json")
                command = f'''python3 /raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/grading.py \
    --system_prompt_path /raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/dpo_sys_prompt.txt \
    --context_path {lab_path}/modified_ps.txt \
    --lab_folder_path {lab_path} \
    --device "{device}" \
    --criterion_name "{key}" \
    --output_file_path {output_path}\
    --parent_dir {parent_dir}\
    --max_length 4096 \
    --adapter_path /raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_models/model18/checkpoint-590 \
'''
    #
                if (lab in eval_labs):
                    command += "--train_split 0.0\n"
                else:
                    command += "--train_split 0.7\n"

                c1 += 1

                if (c1 == 3):
                    c1 = 0
                    c2 += 1

                if (c2 == len(devices)):
                    c2 = 0

                file_name = f"dpo_grade{file_no}.sh"
                file_path = os.path.join(
                    "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_scripts/model18scnewonunscrambled", file_name)

                if (first_time[file_no - 1] == 0):
                    first_time[file_no - 1] = 1
                    with open(file_path, "w") as f:
                        f.write("\n")

                file_no = (file_no + 1) % (num_files + 1)

                if (file_no == 0):
                    file_no = 1

                with open(file_path, "a") as f:
                    f.write(command)
