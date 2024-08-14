import json
import os
import csv
from dataset_utils import *


def get_rubrics(course_code):
    json_file = os.path.join(course_code, "rubrics.json")
    with open(json_file, 'r') as f:
        rubrics_data = json.load(f)

    parsed_rubrics = {}
    for item in rubrics_data:
        title = item['title']
        description = item['description']
        ratings = {rating['title']: rating['description']
                   for rating in item['Ratings']}
        parsed_rubrics[title] = {
            'description': description, 'ratings': ratings}

    return parsed_rubrics


def extract_all_original_grades(grades_file_path):
    '''
    A dictionary with criteria as keys. The value for each criterion will be another dictionary with student ids as keys and grades as values
    '''
    with open(grades_file_path, "r") as f:
        reader = csv.reader(f)

        # All the rows in the csv
        rows = []
        for row in reader:
            rows.append(row)

        # Start and end indices for each criterion
        criterion_indices = {}

        current_criterion = ""
        for i in range(1, len(rows[0])):
            if (not current_criterion):
                current_criterion = rows[0][i]
                start_idx = i
            elif (current_criterion != rows[0][i]):
                end_idx = i - 1
                criterion_indices[current_criterion] = [start_idx, end_idx]
                current_criterion = rows[0][i]
                start_idx = i

        criterion_indices[current_criterion] = [start_idx, len(rows[0]) - 1]

        criterion_rating_titles = {}
        for criterion in criterion_indices.keys():
            start_idx = criterion_indices[criterion][0]
            end_idx = criterion_indices[criterion][1]

            rating_titles = []
            for idx in range(start_idx, end_idx + 1):
                rating_titles.append(rows[2][idx])

            criterion_rating_titles[criterion] = rating_titles

        grades = {}
        for criterion in criterion_indices.keys():
            grades[criterion] = {}
            start_idx = criterion_indices[criterion][0]
            end_idx = criterion_indices[criterion][1]

            for i in range(6, len(rows)):
                student_id = rows[i][0]
                student_id = student_id.split('@')[0]

                for idx in range(start_idx, end_idx + 1):
                    if (rows[i][idx] == '1'):
                        grades[criterion][student_id] = criterion_rating_titles[criterion][idx - start_idx]

        return grades


def extract_original_criterion_grade(grades_file_path, criterion_name):
    grades_file_path = os.path.join(grades_file_path, 'rubric_ratings.csv')
    all_grades = extract_all_original_grades(grades_file_path)

    return all_grades[criterion_name]


def extract_llm_ratings(lab_results_path, criterion_name, criterion_res_path=""):
    predicted_results = {}
    if (criterion_res_path):
        criterion_results_path = criterion_res_path
    else:
        criterion_name = criterion_name.strip()
        criterion_results_path = os.path.join(
            lab_results_path, f'{criterion_name}.json')

    prefix = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/zero_shot_results/"
    # Remove the prefix
    relative_path = criterion_results_path.replace(prefix, "")
    # print(relative_path)

    # LLM outputss

    if (os.path.isfile(criterion_results_path)):
        with open(criterion_results_path, "r") as f:
            data = json.load(f)
        count = 0
        for student_id, model_response in data.items():
            stripped_model_response = model_response.strip()
            start_index = model_response.find('{')
            end_index = model_response.find('}') + 1

            content_within_braces = model_response[start_index:end_index]

            already_extracted = 1
            try:
                # print(criterion_results_path)
                extracted_ans = json_from_string(content_within_braces)
                already_extracted = 0
            except:
                if (stripped_model_response.startswith('''{\n"answer": "''')):
                    # print(stripped_model_response[0], 1111)
                    option = stripped_model_response[13]
                elif (stripped_model_response.startswith('''{\"answer\" : ''')):
                    option = stripped_model_response[12]
                elif (stripped_model_response.startswith('''{\n    \"answer\": \"''')):
                    # print(stripped_model_response[17])
                    option = stripped_model_response[17]
                elif (stripped_model_response.startswith("The correct answer is ")):
                    option = stripped_model_response[22]
                elif (stripped_model_response.startswith("Answer: ")):
                    option = stripped_model_response[8]
                elif (len(stripped_model_response) > 1 and stripped_model_response[1] == "."):
                    # print(stripped_model_response[:30])
                    option = stripped_model_response[0]
                else:
                    count += 1
                    # print(student_id, model_response)
                    continue

            if not (already_extracted):
                try:
                    option = extracted_ans['answer'][0]
                except Exception as e:
                    continue
                # try:
                #     reasoning = extracted_ans['reasoning']
                # except Exception as e:
                #     continue

            try:
                option = option.capitalize()
            except Exception as e:
                pass

            diff = ord(option) - ord('A')
            if not (diff >= 0 and diff < 4):
                # print(student_id, model_response[:20])
                continue
            # result=[]
            # result.append(option)
            # result.append(reasoning)
            # print(result)
            predicted_results[student_id] = option

        return predicted_results
    else:
        return None


def calculate_metrics(original_grades, model_grades):
    '''
    Calculates metrics like accuracy by comparing TA grades and grades assigned by model
    '''
    correct = 0
    one_off_correct = 0

    higher_by_model = 0   # Model assigns a better grade
    lower_by_model = 0    # Model assigns a worse grade

    tp = {}
    fp = {}
    tn = {}
    fn = {}

    for student_id in original_grades.keys():
        tp[original_grades[student_id].strip()] = 0
        fp[original_grades[student_id].strip()] = 0
        tn[original_grades[student_id].strip()] = 0
        fn[original_grades[student_id].strip()] = 0

    for student_id in model_grades.keys():
        tp[model_grades[student_id].strip()] = 0
        fp[model_grades[student_id].strip()] = 0
        tn[model_grades[student_id].strip()] = 0
        fn[model_grades[student_id].strip()] = 0

    total = 0
    for student_id in model_grades.keys():
        if (student_id in original_grades.keys()):
            model_grade = ord(model_grades[student_id].strip()) - ord('A')
            original_grades[student_id] = original_grades[student_id].strip()

            if (len(original_grades[student_id]) != 1):
                continue
            original_grade = ord(original_grades[student_id]) - ord('A')

            total += 1
            if (model_grade == original_grade):
                correct += 1
                tp[model_grades[student_id].strip()] += 1
                for grade in tp.keys():
                    if (grade != model_grades[student_id].strip()):
                        tn[grade] += 1
            else:
                fp[model_grades[student_id].strip()] += 1
                fn[original_grades[student_id].strip()] += 1
                for grade in tp.keys():
                    if ((grade != model_grades[student_id].strip()) and (grade != original_grades[student_id].strip())):
                        tn[grade] += 1

            if (abs(model_grade - original_grade) <= 1):
                one_off_correct += 1

            if (model_grade > original_grade):
                lower_by_model += 1

            if (model_grade < original_grade):
                higher_by_model += 1

    if total == 0:
        print("THIS LAB Output format generated by model is invalid")
        return

    # print(f"Total number of submissions : {total}")
    precisions = {}
    recalls = {}

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for grade in tp.keys():
        total_tp += tp[grade]
        total_fp += fp[grade]
        total_tn += tn[grade]
        total_fn += fn[grade]

        # print(grade, tp[grade], fp[grade], tn[grade], fn[grade])

        if ((tp[grade] + fp[grade] > 0) and (tp[grade] + fn[grade] > 0)):
            precisions[grade] = tp[grade]/(tp[grade] + fp[grade])
            recalls[grade] = tp[grade]/(tp[grade] + fn[grade])

        if ((tp[grade] + fp[grade] == 0) and fn[grade] > 0):
            precisions[grade] = 0
            recalls[grade] = 0

        if ((tp[grade] == 0) and (fp[grade] == 0) and (fn[grade] == 0)):
            precisions[grade] = 1
            recalls[grade] = 1

    count = 0
    avg_precision = 0
    avg_recall = 0
    for grade in precisions.keys():
        count += 1
        avg_precision += precisions[grade]
        avg_recall += recalls[grade]
    if (count > 0):
        avg_precision = avg_precision/count
        avg_recall = avg_recall/count
        # print(f"Avg precision : {avg_precision}")
        # print(f"Avg recall : {avg_recall}")

    # print(f"Accuracy : {correct/total}")
    # print(f"Total : {total}\n")
    # print(f"One off agreement : {one_off_correct/total}")
    # print(f"Percentage of cases where model assigns better grade : {(higher_by_model*100)/total}")
    # print(f"Percentage of cases where model assigns worse grade : {(lower_by_model*100)/total}")

    # avg_precision = (total_tp)/(total_tp + total_fp)
    # avg_recall = (total_tp)/(total_tp + total_fn)

    # print(f"Avg precision : {avg_precision}")
    # print(f"Avg recall : {avg_recall}")

    json_obj = {"count": total, "accuracy": correct/total,
                "precision": avg_precision, "recall": avg_recall, "total": total}
    return json_obj


def extract_results():
    directory = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/dataset/"
    results_par_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/"
    labs_to_consider = ["CP_00105_sort_cs101f22_LE2_E_Q2", "CP_00108_loop_cs101f22_LE01_B_Q2",
                        "CP_00113_sort_cs101f22_LE03_A_Q1", "CP_00102_loop_cs101f22_LE01_A_Q2"]
    zs_folder_name = "zero_shot_results"
    zs_path = os.path.join(results_par_folder, zs_folder_name)

    accuracy_dict = {}
    for lab in labs_to_consider:
        # print(f'Lab Name: {lab}')
        lab_path = os.path.join(directory, lab)
        rubrics = get_rubrics(lab_path)
        criteria_to_consider = []
        for criterion in rubrics.keys():
            criteria_to_consider.append(criterion)
        # print(f'No of Criterions : {len(criteria_to_consider)}')
        lab_accuracy_dict = {}  # Inner dictionary for current lab
        for criterion in criteria_to_consider:
            original_grds = extract_original_criterion_grade(
                lab_path, criterion)
            zs_grds = extract_llm_ratings(
                os.path.join(zs_path, lab), criterion)
            # print(original_grds)
            # print(zs_grds)
            ans = calculate_metrics(original_grds, zs_grds)
            lab_accuracy_dict[criterion] = ans['accuracy']
        accuracy_dict[lab] = lab_accuracy_dict
        # print("***************************Done******************************************")
        # print()
        # print()
        # print()
    print(accuracy_dict)


def extract_zs_fs_results():
    directory = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset24"
    results_path_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/"
    labs_to_consider = ["CP_00105_sort_cs101f22_LE2_E_Q2", "CP_00108_loop_cs101f22_LE01_B_Q2",
                        "CP_00113_sort_cs101f22_LE03_A_Q1", "CP_00102_loop_cs101f22_LE01_A_Q2", 'cs101s23_lq01_c_q4', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q4', 'cs101a23_lq01_d_q3']
    # labs_to_consider = ['CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1', 'CP_00113_sort_cs101f22_LE03_A_Q1',
    #                     'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00208_var_LQ01_B_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00216_var_cs101s23_LQ01_D_Q4', 'CP_00315_var_cs101a23_lq01_c_q5', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q3', 'cs101a23_lq01_d_q4', 'cs101s23_lq01_c_q4']
    zs_folder_name = "zero_shot_results"
    fs_folder_name = "few_shot_results_verbose"
    zs_path = os.path.join(results_path_folder, zs_folder_name)
    fs_path = os.path.join(results_path_folder, fs_folder_name)
    # dpo_path = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model_1"
    zs_accuracy_dict = {}
    fs_accuracy_dict = {}
    zs_p_dict = {}
    fs_p_dict = {}
    zs_r_dict = {}
    fs_r_dict = {}

    zs_test_numbers = {}
    # dpo_accuracy_dict = {}

    for lab in labs_to_consider:

        lab_path = os.path.join(directory, lab)
        rubrics = get_rubrics(lab_path)
        criteria_to_consider = []
        for criterion in rubrics.keys():
            criteria_to_consider.append(criterion)

        zs_lab_accuracy_dict = {}
        fs_lab_accuracy_dict = {}
        zs_lab_p_dict = {}
        fs_lab_p_dict = {}
        zs_lab_r_dict = {}
        fs_lab_r_dict = {}
        # dpo_lab_accuracy_dict = {}  # Inner dictionary for current lab
        for criterion in criteria_to_consider:
            original_grds = extract_original_criterion_grade(
                lab_path, criterion)
            zs_grds = extract_llm_ratings(
                os.path.join(zs_path, lab), criterion)
            fs_grds = extract_llm_ratings(
                os.path.join(fs_path, lab), criterion)
            # dpo_grds = extract_llm_ratings(os.path.join(dpo_path, lab), criterion)
            zs_no_grades = jsonitems(os.path.join(zs_path, lab), criterion)
            fs_no_grades = jsonitems(os.path.join(fs_path, lab), criterion)
            print(len(original_grds), len(zs_grds),
                  zs_no_grades, lab, criterion)
            zs_ans = calculate_metrics(original_grds, zs_grds)
            fs_ans = calculate_metrics(original_grds, fs_grds)
            # dpo_ans = calculate_metrics(original_grds, dpo_grds)
            zs_lab_accuracy_dict[criterion] = zs_ans['accuracy']
            fs_lab_accuracy_dict[criterion] = fs_ans['accuracy']
            zs_lab_p_dict[criterion] = zs_ans['precision']
            fs_lab_p_dict[criterion] = fs_ans['precision']
            zs_lab_r_dict[criterion] = zs_ans['recall']
            fs_lab_r_dict[criterion] = fs_ans['recall']

            # dpo_lab_accuracy_dict[criterion] = dpo_ans['accuracy']
        zs_accuracy_dict[lab] = zs_lab_accuracy_dict
        fs_accuracy_dict[lab] = fs_lab_accuracy_dict
        zs_p_dict[lab] = zs_lab_p_dict
        fs_p_dict[lab] = fs_lab_p_dict
        zs_r_dict[lab] = zs_lab_r_dict
        fs_r_dict[lab] = fs_lab_r_dict
        zs_test_numbers[lab] = len(zs_grds)
        # dpo_accuracy_dict[lab] = dpo_lab_accuracy_dict
        # print("***************************Done******************************************")
        # print()
        # print()
        # print()
    # print(zs_accuracy_dict)
    print()
    # print(fs_accuracy_dict)
    zs_macro = macro_accuracy(zs_accuracy_dict)
    fs_macro = macro_accuracy(fs_accuracy_dict)
    zs_macro_p = macro_accuracy(zs_p_dict)
    fs_macro_p = macro_accuracy(fs_p_dict)
    zs_macro_r = macro_accuracy(zs_r_dict)
    fs_macro_r = macro_accuracy(fs_r_dict)

    zs_overall = micro_overall_accuracy(zs_accuracy_dict, zs_test_numbers)
    fs_overall = micro_overall_accuracy(fs_accuracy_dict, zs_test_numbers)
    zs_overallp = micro_overall_accuracy(zs_p_dict, zs_test_numbers)
    fs_overallp = micro_overall_accuracy(fs_p_dict, zs_test_numbers)
    zs_overallr = micro_overall_accuracy(zs_r_dict, zs_test_numbers)
    fs_overallr = micro_overall_accuracy(fs_r_dict, zs_test_numbers)
    # dpo_macro = macro_accuracy(dpo_accuracy_dict)
    print()
    print()
    # print(zs_macro)
    print(f"{zs_overall:.3f} {fs_overall:.3f} {zs_overallp:.3f} {fs_overallp:.3f} {zs_overallr:.3f} {fs_overallr:.3f}")
    # print(fs_macro)
    # print(dpo_macro)
    # print(dpo_accuracy_dict)


def macro_accuracy(accuracy_dict):
    labs = list(accuracy_dict.keys())
    macro_acc = {}
    for lab in labs:
        lab_accuracy_dict = accuracy_dict[lab]
        all_accuracies = list(lab_accuracy_dict.values())
        avg_acc = sum(all_accuracies)/len(all_accuracies)
        macro_acc[lab] = avg_acc
    return macro_acc


def macro_overall_accuracy(lab_accuracy_dict):
    accuracies = list(lab_accuracy_dict.values())
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy


def micro_overall_accuracy(accuracy_dict, test_numbers):
    macro_acc = macro_accuracy(accuracy_dict)
    labs = list(macro_acc.keys())
    overall_acc = 0
    cumulative_accuracy = 0
    total_nos = 0
    for lab in labs:
        cumulative_accuracy += macro_acc[lab]*test_numbers[lab]
        total_nos += test_numbers[lab]
    overall_acc = cumulative_accuracy/total_nos
    return overall_acc


# extract_zs_fs_results()


def dpo_results(results_path, labs):
    labs_to_consider = labs
    dpo_path = results_path
    directory = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27"
    dpo_accuracy_dict = {}
    dpo_p_dict = {}
    dpo_r_dict = {}
    dpo_test_numbers = {}
    for lab in labs_to_consider:
        lab_path = os.path.join(directory, lab)
        rubrics = get_rubrics(lab_path)
        criteria_to_consider = []
        for criterion in rubrics.keys():
            criteria_to_consider.append(criterion)
        dpo_lab_accuracy_dict = {}
        dpo_lab_p_dict = {}
        dpo_lab_r_dict = {}
        for criterion in criteria_to_consider:
            original_grds = extract_original_criterion_grade(
                lab_path, criterion)

            dpo_grds = extract_llm_ratings(
                os.path.join(dpo_path, lab), criterion)
            llm_no_grades = jsonitems(os.path.join(dpo_path, lab), criterion)
            print(len(original_grds), len(dpo_grds),
                  llm_no_grades, lab, criterion)
            # print(dpo_grds, 64654564, lab, criterion)
            dpo_ans = calculate_metrics(original_grds, dpo_grds)
            dpo_lab_accuracy_dict[criterion] = dpo_ans['accuracy']
            dpo_lab_p_dict[criterion] = dpo_ans['precision']
            dpo_lab_r_dict[criterion] = dpo_ans['recall']

            # print(dpo_grds, criterion, lab, 123123123123)
        dpo_accuracy_dict[lab] = dpo_lab_accuracy_dict
        dpo_p_dict[lab] = dpo_lab_p_dict
        dpo_r_dict[lab] = dpo_lab_r_dict
        dpo_test_numbers[lab] = len(dpo_grds)
    dpo_macro = macro_accuracy(dpo_accuracy_dict)
    dpo_macro_overall = macro_overall_accuracy(dpo_macro)
    dpo_overall = micro_overall_accuracy(dpo_accuracy_dict, dpo_test_numbers)
    dpo_overallp = micro_overall_accuracy(dpo_p_dict, dpo_test_numbers)
    dpo_overallr = micro_overall_accuracy(dpo_r_dict, dpo_test_numbers)
    dpo_overallf = 2 * dpo_overallp * dpo_overallr/(dpo_overallp+dpo_overallr)
    print()
    print(f"Micro overall Accuracy:{dpo_overall:.4f}")
    print(f"Precision:{dpo_overallp:.4f}")
    print(f"Recall:{dpo_overallr:.4f}")
    print(f"F1score:{dpo_overallf:.4f}")
    print(
        f"Macro overall Accuracy:{dpo_macro_overall:.4f}             Averaged across labs(classes)")
    # print(zs_macro)
    print()
    # print(fs_macro)
    print(dpo_macro)
    print(dpo_accuracy_dict)


def jsonitems(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')
    with open(criterion_results_path, 'r') as f:
        data = json.load(f)
    return len(data)


def dpo_results_unseen_common(results_path, labs, eval_labs, unseen, common):
    # if n = 1: it will be unseen labs(test), else seen labs
    labs_to_consider = labs
    dpo_path = results_path
    directory = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27"
    dpo_accuracy_dict = {}
    dpo_p_dict = {}
    dpo_r_dict = {}
    dpo_test_numbers = {}
    for lab in labs_to_consider:
        if (lab in eval_labs) if unseen else (lab not in eval_labs):
            lab_path = os.path.join(directory, lab)
            rubrics = get_rubrics(lab_path)
            allc, commonc, uncommonc = create_common_criteria()
            criteria_to_consider = []
            if common:
                criteria_to_consider = commonc[lab]
            else:
                criteria_to_consider = uncommonc[lab]
            # for criterion in rubrics.keys():
            #     criteria_to_consider.append(criterion)
            dpo_lab_accuracy_dict = {}
            dpo_lab_p_dict = {}
            dpo_lab_r_dict = {}
            for criterion in criteria_to_consider:
                original_grds = extract_original_criterion_grade(
                    lab_path, criterion)
                dpo_grds = extract_llm_ratings(
                    os.path.join(dpo_path, lab), criterion)
                llm_no_grades = jsonitems(
                    os.path.join(dpo_path, lab), criterion)
                print(len(original_grds), len(dpo_grds),
                      llm_no_grades, lab, criterion)
                # print(dpo_grds, 64654564, lab, criterion)
                dpo_ans = calculate_metrics(original_grds, dpo_grds)
                dpo_lab_accuracy_dict[criterion] = dpo_ans['accuracy']
                dpo_lab_p_dict[criterion] = dpo_ans['precision']
                dpo_lab_r_dict[criterion] = dpo_ans['recall']

                # print(dpo_grds, criterion, lab, 123123123123)
            dpo_accuracy_dict[lab] = dpo_lab_accuracy_dict
            dpo_p_dict[lab] = dpo_lab_p_dict
            dpo_r_dict[lab] = dpo_lab_r_dict
            dpo_test_numbers[lab] = len(dpo_grds)
    dpo_macro = macro_accuracy(dpo_accuracy_dict)
    dpo_macro_overall = macro_overall_accuracy(dpo_macro)
    dpo_overall = micro_overall_accuracy(dpo_accuracy_dict, dpo_test_numbers)
    dpo_overallp = micro_overall_accuracy(dpo_p_dict, dpo_test_numbers)
    dpo_overallr = micro_overall_accuracy(dpo_r_dict, dpo_test_numbers)
    dpo_overallf = 2 * dpo_overallp * dpo_overallr/(dpo_overallp+dpo_overallr)
    print()
    print(f"Micro overall Accuracy:{dpo_overall:.4f}")
    print(f"Precision:{dpo_overallp:.4f}")
    print(f"Recall:{dpo_overallr:.4f}")
    print(f"F1score:{dpo_overallf:.4f}")
    print(f"Macro overall Accuracy:{dpo_macro_overall:.4f}")
    # print(zs_macro)
    print()
    # print(fs_macro)
    print(dpo_macro)
    print(dpo_accuracy_dict)


def inspect_results(folder_name):
    '''
    Given the path to a results folder, we print rating wise accuracy, precision and recall metrics

    Args : 
        folder_name (str) : Path of the folder where the results are stored

    Returns : 
        None
    '''
    directory = "/raid/ganesh/nagakalyani/nagakalyani/ICER/dataset/"
    results_par_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/"

    results_path = os.path.join(results_par_folder, folder_name)

    llm_grades = {}
    og_grades = {}

    for lab in os.listdir(directory):
        if (lab.startswith("CP")):
            # print(lab)
            lab_path = os.path.join(directory, lab)
            rubrics = get_rubrics(lab_path)

            llm_grades[lab] = {}
            og_grades[lab] = {}

            for criterion in rubrics.keys():
                # print(f'''{criterion} : {rubrics[criterion]["description"]}''')
                original_grades = extract_original_criterion_grade(
                    lab_path, criterion)
                grades = extract_llm_ratings(
                    os.path.join(results_path, lab), criterion)

                if (grades):
                    llm_grades[lab][criterion] = grades
                    og_grades[lab][criterion] = original_grades

                # if (grades) :
                #     calculate_metrics(original_grades, grades)

    criterion_wise_accuracies, lab_wise_micro_accuracies, lab_wise_macro_accuracies, criterion_wise_macro_accuracy, lab_wise_macro_accuracy, overall_accuracy = calculate_accuracy_metrics(
        og_grades, llm_grades)
    tp, fp, tn, fn, rating_wise_metrics, criterion_wise_metrics, lab_wise_metrics, overall_rating_wise_metrics, overall_criterion_wise_metrics, overall_lab_wise_metrics, original_rating_counts, model_rating_counts = calculate_precision_recall(
        og_grades, llm_grades)

    lab_names = criterion_wise_accuracies.keys()
    for lab_name in lab_names:
        print(lab_name)
        for criterion_name in criterion_wise_accuracies[lab_name].keys():
            possible_grades = model_rating_counts[lab_name][criterion_name].keys(
            )
            print(criterion_name)

            print("Original grades : ", end="")
            for grade in sorted(possible_grades):
                print(
                    f"{grade} : {original_rating_counts[lab_name][criterion_name][grade]}", end=" ")
            print("")

            print("Model grades : ", end="")
            for grade in sorted(possible_grades):
                print(
                    f"{grade} : {model_rating_counts[lab_name][criterion_name][grade]}", end=" ")
            print("")

            print("Rating wise metrics : ")
            print("Precision :")
            for grade in sorted(possible_grades):
                precis = rating_wise_metrics[lab_name][criterion_name][grade]["precision"]
                print(f"{grade} : {precis}", end=" ")
            print("")
            print("Recall")
            for grade in sorted(possible_grades):
                rec = rating_wise_metrics[lab_name][criterion_name][grade]["recall"]
                print(f"{grade} : {rec}", end=" ")
            print("")

            print("Accuracy :", criterion_wise_accuracies[lab_name][criterion_name], "Precision :", criterion_wise_metrics[lab_name]
                  [criterion_name]["precision"], "Recall :", criterion_wise_metrics[lab_name][criterion_name]["recall"])
        print("")


def create_common_criteria(dataset_path="/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27"):
    criteria = {}
    common_criteria = {}
    uncommon_criteria = {}
    labs_new = ['CP_00102_loop_cs101f22_LE01_A_Q2', 'cs101a23_lq01_a_q3', 'CP_00105_sort_cs101f22_LE2_E_Q2',
                'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2']
    labs_new2 = ['cs101a23_lq01_a_q2', 'cs101a23_lq01_a_q5', 'cs101a23_lq01_b_q4',
                 'cs101a23_lq01_d_q3', 'cs101s23_lq01_c_q4',  'cs101a23_lq01_d_q4']
    for lab in os.listdir(dataset_path):
        lab_path = os.path.join(dataset_path, lab)
        rubric = get_rubrics(lab_path)
        criteria[lab] = []
        common_criteria[lab] = []
        uncommon_criteria[lab] = []
        for criterion in rubric.keys():
            criteria[lab].append(criterion)
        if lab in labs_new:
            for criterion in rubric.keys():
                if int(criterion[-2:]) <= 6 and int(criterion[-1]) != 2:
                    common_criteria[lab].append(criterion)
                else:
                    uncommon_criteria[lab].append(criterion)
        elif lab in labs_new2:
            n = len(rubric)
            if n == 14:
                n = n-1
            if n == 7:
                n = n+1
            for criterion in rubric.keys():
                if n-int(criterion[-2:]) <= 4 and n-int(criterion[-2:]) >= 0:
                    common_criteria[lab].append(criterion)
                else:
                    uncommon_criteria[lab].append(criterion)
        else:
            for criterion in rubric.keys():
                if len(criterion) == 11 or len(criterion) == 12 or len(criterion) == 10 or criterion == 'Implementation':
                    uncommon_criteria[lab].append(criterion)
                else:
                    common_criteria[lab].append(criterion)

    return criteria, common_criteria, uncommon_criteria


if __name__ == "__main__":
    labs_new = ['cs101a23_lq01_a_q2', 'cs101a23_lq01_a_q5', 'cs101a23_lq01_b_q4', 'cs101a23_lq01_d_q3', 'cs101s23_lq01_c_q4',  'cs101a23_lq01_d_q4',
                'CP_00102_loop_cs101f22_LE01_A_Q2', 'cs101a23_lq01_a_q3', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2']
    # labs_old = ['CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1', 'CP_00315_var_cs101a23_lq01_c_q5', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00216_var_cs101s23_LQ01_D_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00208_var_LQ01_B_Q4']
    # extract_zs_fs_results()
    labs27 = ['cs101a23_lq01_a_q2', 'cs101a23_lq01_a_q5', 'cs101a23_lq01_b_q4', 'CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1',
              'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00216_var_cs101s23_LQ01_D_Q4', 'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00208_var_LQ01_B_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00315_var_cs101a23_lq01_c_q5', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q3', 'cs101a23_lq01_d_q4', 'cs101s23_lq01_c_q4']
    labs26 = ['cs101a23_lq01_a_q2', 'cs101a23_lq01_a_q5', 'cs101a23_lq01_b_q4', 'CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1',
              'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00208_var_LQ01_B_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00315_var_cs101a23_lq01_c_q5', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q3', 'cs101a23_lq01_d_q4', 'cs101s23_lq01_c_q4']
    labs23 = ['CP_00101_loop_cs101f22_LE01_A_Q1', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'CP_00104_loop_cs101f22_LE02_B_Q1', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00106_loop_cs101f22_LE2_A_Q2', 'CP_00107_sort_cs101f22_LE03_B_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2', 'CP_00110_loop_cs101f22_LE01_C_Q2', 'CP_00111_array_cs101f22_LE02_B_Q2', 'CP_00112_fun_cs101f22_LE02_C_Q1',
              'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00115_sort_cs101f22_LE03_C_Q1', 'CP_00206_var_cs101s23_lq01_b_q2', 'CP_00116_sort_cs101f22_LE03_E_Q1', 'CP_00117_mat_cs101f22_LE03_B_Q2', 'CP_00208_var_LQ01_B_Q4', 'CP_00211_condition_cs101s23_LQ01_C_Q3', 'CP_00214_var_cs101s23_lq01_d_q2', 'CP_00315_var_cs101a23_lq01_c_q5', 'cs101a23_lq01_a_q3', 'cs101a23_lq01_d_q3', 'cs101a23_lq01_d_q4', 'cs101s23_lq01_c_q4']
    eval_labs18 = [
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
    eval_labs18high = [
        "CP_00102_loop_cs101f22_LE01_A_Q2",
        "CP_00214_var_cs101s23_lq01_d_q2",
        "cs101a23_lq01_b_q4",
        "CP_00106_loop_cs101f22_LE2_A_Q2",
        "cs101a23_lq01_a_q3",
        "cs101a23_lq01_a_q2",
        "CP_00108_loop_cs101f22_LE01_B_Q2",
        "CP_00211_condition_cs101s23_LQ01_C_Q3",
        "cs101a23_lq01_d_q3"
    ]
    eval_labstop6 = [
        'CP_00214_var_cs101s23_lq01_d_q2',
        'cs101a23_lq01_a_q2',
        'CP_00108_loop_cs101f22_LE01_B_Q2',
        'CP_00211_condition_cs101s23_LQ01_C_Q3',
        'cs101a23_lq01_d_q3',
        'CP_00206_var_cs101s23_lq01_b_q2'
    ]
    evalnone = []
    eval_labs3 = ["CP_00104_loop_cs101f22_LE02_B_Q1", "CP_00107_sort_cs101f22_LE03_B_Q1", "CP_00112_fun_cs101f22_LE02_C_Q1", "CP_00116_sort_cs101f22_LE03_E_Q1",
                  "CP_00211_condition_cs101s23_LQ01_C_Q3"]  # , "CP_00315_var_cs101a23_lq01_c_q5", "CP_00117_mat_cs101f22_LE03_B_Q2", "CP_00110_loop_cs101f22_LE01_C_Q2"]
    eval_labs2 = ["cs101a23_lq01_a_q3", "CP_00105_sort_cs101f22_LE2_E_Q2", "cs101s23_lq01_c_q4",
                  "CP_00111_array_cs101f22_LE02_B_Q2", "CP_00108_loop_cs101f22_LE01_B_Q2"]
    eval_labs1 = ["CP_00101_loop_cs101f22_LE01_A_Q1", "CP_00106_loop_cs101f22_LE2_A_Q2",
                  "CP_00115_sort_cs101f22_LE03_C_Q1", "CP_00214_var_cs101s23_lq01_d_q2", "cs101a23_lq01_d_q4"]
    # , 'CP_00216_var_cs101s23_LQ01_D_Q4'
    # dpo_results(
    #     "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18", labs26)
    # dpo_results_unseen_common(
    #     "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/modeltop6", labs26, eval_labstop6, 0, 0)
    # dpo_results(
    #     "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18sconunscrambled", labs26)
    results_path = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model18scnewonsc"
    dpo_results_unseen_common(results_path, labs26, eval_labs18, 0, 1)
    dpo_results_unseen_common(results_path, labs26, eval_labs18, 0, 0)
    dpo_results_unseen_common(results_path, labs26, eval_labs18, 1, 1)
    dpo_results_unseen_common(results_path, labs26, eval_labs18, 1, 0)

    # inspect_results("/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dpo_results/model_1")
    # a = get_rubrics("/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27/CP_00101_loop_cs101f22_LE01_A_Q1")
    a1, b1, c1 = create_common_criteria()
    a = {key: a1[key] for key in labs_new +
         [key for key in a1 if key not in labs_new]}
    b = {key: b1[key] for key in labs_new +
         [key for key in b1 if key not in labs_new]}
    c = {key: c1[key] for key in labs_new +
         [key for key in c1 if key not in labs_new]}
    # print(a)
    # print()
    # print(b)
    # for key in b.keys():
    #     print(c[key], key)
    #     print()
    # print(c)
    total_length = sum(len(value) for value in b.values())
    total_lengtha = sum(len(value) for value in a.values())

    print("Total common criteria: ", total_length)
    print("Total criteria: ", total_lengtha)
