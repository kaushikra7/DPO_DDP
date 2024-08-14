
import random

seed = 42
random.seed(seed)
labs = ['CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1', 'CP_00113_sort_cs101f22_LE03_A_Q1',
        'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00208_var_LQ01_B_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00216_var_cs101s23_LQ01_D_Q4', 'CP_00315_var_cs101a23_lq01_c_q5', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q3', 'cs101a23_lq01_d_q4', 'cs101s23_lq01_c_q4']

selected_elements = random.sample(labs, 5)

print(selected_elements)
