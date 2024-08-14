import os
import json
import sys
import csv
import ast
import random
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def json_from_string(string):
    return ast.literal_eval(string.strip())


def get_submission_json(course_code):
    '''
    Returns all the student submissions for a given problem statement

    Args :
        course_code (str) : The path to the submissions for that lab

    Returns : 
        submissions_data (dict) : A dictionary of student submissions. Student IDs are the keys
    '''
    submissions_dir = os.path.join(course_code, "submissions")
    submissions_data = {}

    # Traverse through each submission directory
    for submission_dir in os.listdir(submissions_dir):
        submission_path = os.path.join(submissions_dir, submission_dir)
        if os.path.isdir(submission_path):
            submission_files = os.listdir(submission_path)
            for file_name in submission_files:
                # Assuming all submissions are C++ files
                if file_name.endswith(".cpp"):
                    file_path = os.path.join(submission_path, file_name)
                    with open(file_path, 'r') as file:
                        submission_code = file.read()
                    submission_key = submission_dir.split('@')[0]
                    submissions_data[submission_key] = submission_code.strip()
                    break  # Assuming there's only one .cpp file per submission

    return submissions_data

    # Write submissions data to JSON file
    # with open(f"{course_code}_submissions.json", 'w') as json_file:
    #     json.dump(submissions_data, json_file, indent=4)


def get_rubrics(course_code):
    '''
    Fetches all the rubrics and corresponding descriptions associated with a problem statement

    Args : 
        course_code (str) : The path to the submissions for the problem statement

    Returns : 
        parsed_rubrics (dict) : A dictionary of rubric titles, descriptions and ratings.
    '''
    json_file = os.path.join(course_code, "rubrics.json")
    # json_file = os.path.join(course_code, "rubrics1.json")
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


def get_criterion_specified_rubrics(course_code, criterion):
    return get_rubrics(course_code)[criterion]['ratings']


def make_folders(parent_dir, output_dir):
    '''
    Creates files with the same names in output_dir as the ones in parent_dir
    '''
    for lab in os.listdir(parent_dir):
        os.mkdir(os.path.join(output_dir, lab))


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
                    if (rows[i][idx] != '0'):
                        grades[criterion][student_id] = criterion_rating_titles[criterion][idx - start_idx]

        return grades


def extract_original_criterion_grade(grades_file_path, criterion_name):
    grades_file_path = os.path.join(grades_file_path, 'rubric_ratings.csv')
    all_grades = extract_all_original_grades(grades_file_path)

    return all_grades[criterion_name]


def extract_all_original_reasonings(grades_file_path):
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

        reasonings = {}
        for criterion in criterion_indices.keys():
            reasonings[criterion] = {}
            start_idx = criterion_indices[criterion][0]
            end_idx = criterion_indices[criterion][1]

            for i in range(6, len(rows)):
                student_id = rows[i][0]
                student_id = student_id.split('@')[0]

                for idx in range(start_idx, end_idx + 1):
                    if (rows[i][idx] != '0'):

                        reasonings[criterion][student_id] = rows[i][idx]
                        if rows[i][idx] == 'No Comments':
                            reasonings[criterion][student_id] = rows[3][idx]
                        # reasonings[criterion][student_id] = criterion_rating_titles[criterion][idx - start_idx]

        return reasonings


def extract_original_criterion_reasoning(grades_file_path, criterion_name):
    grades_file_path = os.path.join(grades_file_path, 'rubric_ratings.csv')
    all_grades = extract_all_original_grades(grades_file_path)

    return all_grades[criterion_name]


def extract_llm_ratings(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    predicted_results = {}
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')

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
                extracted_ans = json_from_string(content_within_braces)
                already_extracted = 0
            except:
                if (stripped_model_response.startswith('''{\n"answer": "''')):
                    option = stripped_model_response[13]
                elif (stripped_model_response.startswith('''{\"answer\" : ''')):
                    option = stripped_model_response[12]
                elif (stripped_model_response.startswith("The correct answer is ")):
                    option = stripped_model_response[22]
                elif (stripped_model_response.startswith("Answer: ")):
                    option = stripped_model_response[8]
                else:
                    count += 1
                    # print(student_id, model_response)
                    continue

            if not (already_extracted):
                try:
                    option = extracted_ans['answer'][0]
                except Exception as e:
                    continue

            try:
                option = option.capitalize()
            except Exception as e:
                pass

            diff = ord(option) - ord('A')
            if not (diff >= 0 and diff < 4):
                # print(student_id, model_response[:20])
                continue

            predicted_results[student_id] = option

        return predicted_results
    else:
        return None


def extract_lora_ratings_new(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    predicted_results = {}
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')

    # LLM outputs
    with open(criterion_results_path, "r") as f:
        data = json.load(f)
    for student_id, model_response in data.items():
        start_index = model_response.find('{')
        end_index = model_response.find('}') + 1

        content_within_braces = model_response[start_index:end_index]
        content_within_braces = content_within_braces.strip()

        option = 'A'
        if (len(content_within_braces) > 12):
            option = content_within_braces[12]
        else:
            continue

        diff = ord(option) - ord('A')
        if not (diff >= 0 and diff < 4):
            continue

        predicted_results[student_id] = option

    return predicted_results


def extract_lora_ratings(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    predicted_results = {}
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')

    # LLM outputs
    with open(criterion_results_path, "r") as f:
        data = json.load(f)
    for student_id, model_response in data.items():
        pattern = r'The correct answer is[:\s]+([A-Z])'
        # Search for the pattern in the model response
        match = re.search(pattern, model_response)
        try:
            option = match.group(1)
        except Exception as e:
            continue
        predicted_results[student_id] = option
    return predicted_results


def extract_lora_ratings_sent(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    predicted_results = {}
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')

    # LLM outputs
    with open(criterion_results_path, "r") as f:
        data = json.load(f)
    for student_id, model_response in data.items():
        stripped_model_response = model_response.strip()
        if (stripped_model_response.startswith("The correct answer is")):
            predicted_results[student_id] = stripped_model_response[22]
        else:
            continue
    return predicted_results


def extract_dpo_ratings(lab_results_path, criterion_name):
    criterion_name = criterion_name.strip()
    predicted_results = {}
    criterion_results_path = os.path.join(
        lab_results_path, f'{criterion_name}.json')

    # LLM outputs

    try:
        with open(criterion_results_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        with open("missing_files.txt", "w+") as f:
            f.write(criterion_results_path)
            f.write("\n")
        return

    for student_id, model_response in data.items():
        try:
            option = model_response.strip()[0]

            diff = ord(option) - ord('A')
            if not (diff >= 0 and diff < 4):
                continue
        except Exception as e:
            continue

        predicted_results[student_id] = option
    return predicted_results


def fewshot_criterion_ids(coursecode):
    '''
    input : coursecode is a lab in the dataset having rubric_ratins.csv file

    output : this function gives two dictionaries as output.

    criterion_data : is a dictionary of dictionary with criteria in that lab as key. the value of that key is a dictionary with keys as unique marks for that criterion and students ids who got that mark for that particular criterion

    rating_descriptions : this is also a dictionary of dictionary with Criterion as key. the value is a dictionary with key as marks of the rating and value is the rating description
    '''
    rubric_path = coursecode + "/rubric_ratings.csv"
    df = pd.read_csv(rubric_path, header=None)
    unique_criteria = df.loc[0].unique()
    unique_elements = []  # Initialize an empty list to store unique elements

    for element in df.loc[0].unique():
        unique_elements.append(element)
    unique_elements.remove('Criterion Title')
    criterion_data = {}
    rating_descriptions = {}
    for element in unique_elements:
        criterion_data[element] = {}
        rating_descriptions[element] = {}
    dataframes = {}
    for element in unique_elements:
        # Select columns where row 0 value is 'Student' or the current unique element
        selected_columns = df.loc[:, (df.loc[0] == 'Criterion Title') | (
            df.loc[0] == element)]
        # Store the selected columns in a new DataFrame
        dataframes[element] = selected_columns
        marks = dataframes[element].iloc[4].unique()
        marks = list(marks)
        marks.remove('Rating marks')
        for mark in marks:
            mark = float(mark)
            mark = int(mark)
            criterion_data[element][mark] = []
            rating_descriptions[element][mark] = ''
        for index, row in dataframes[element].iloc[6:].iterrows():
            keys = dataframes[element].iloc[index].keys()
            keys = list(keys)
            keys.remove(0)
            for key in keys:
                mark = dataframes[element].iloc[4][key]
                mark = float(mark)
                mark = int(mark)
                if rating_descriptions[element][mark] == '':
                    rating_descriptions[element][mark] = dataframes[element].iloc[3][key]

                value = dataframes[element].iloc[index][key]
                if dataframes[element].iloc[index][key] == '1':
                    markgiven = dataframes[element].iloc[4][key]
                    markgiven = float(markgiven)
                    markgiven = int(markgiven)
                    criterion_data[element][markgiven].append(
                        dataframes[element].iloc[index][0])

    return criterion_data, rating_descriptions


def fewshot_examples_retriever(shots, coursecode, criterion):
    ids = {}
    total_rubric_data, total_rating_descriptions = fewshot_criterion_ids(
        coursecode)
    criterion_data = total_rubric_data[criterion]
    rating_descriptions = total_rating_descriptions[criterion]
    ratings = len(criterion_data)
    n = shots
    for _ in criterion_data.keys():
        ids[_] = []
    while (n > 0):
        for keys in ids.keys():
            if n == 0:
                break
            for key, value in criterion_data.items():
                if len(criterion_data[key]) > 0:
                    if key == keys:
                        ids[key].append(
                            value[random.randrange(len(criterion_data[key]))])
                        n -= 1
                        break

    descriptions_and_codes = fewshot_codes(
        ids, rating_descriptions, coursecode)
    return ids, descriptions_and_codes


def fewshot_codes(fewshot_ids, rating_descriptions, coursecode):
    '''
    input : fewshot_ids - refer the fewshot_criterion_ids function

    output : it is a dictionary of rating description as key and student codes who got that ratings as value (list of student codes with same rating)

    '''
    idss = {}
    for key in fewshot_ids.keys():
        idss[key] = []
    total_submission_data = get_submission_json(coursecode)
    for key, students in fewshot_ids.items():
        for student in students:
            studentid = student.split('@')[0]
            studentcode = total_submission_data[studentid]
            idss[key].append(studentcode)
    final_ids = {rating_descriptions[old_key]
        : value for old_key, value in idss.items()}
    return final_ids


def calculate_metrics(original_grades, model_grades):
    '''
    Calculates metrics like accuracy, precision and recall by comparing TA grades and grades assigned by model for a particular criterion of a lab

    Args : 
        original_grades (dict) : A dictionary of original TA grades. Student ids are the keys. 
        model_grades (dict) : A dictionary of model assigned grades. Student ids are the keys.

    Returns : 
        A dictionary with fields "accuracy", "precision", "recall" and "total" (No. of submissions for which both grades are available)
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
                print("False negitives model assigned lower grade\n")
                print(student_id)

            if (model_grade < original_grade):
                higher_by_model += 1
                print("False positives model assigned higher grade\n")
                print(student_id)

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

        print(grade, tp[grade], fp[grade], tn[grade], fn[grade])

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
        print(f"Avg precision : {avg_precision}")
        print(f"Avg recall : {avg_recall}")
    print(f"Accuracy : {correct/total}")
    print(f"Total : {total}\n")
    # print(f"One off agreement : {one_off_correct/total}")
    # print(f"Percentage of cases where model assigns better grade : {(higher_by_model*100)/total}")
    # print(f"Percentage of cases where model assigns worse grade : {(lower_by_model*100)/total}")

    # avg_precision = (total_tp)/(total_tp + total_fp)
    # avg_recall = (total_tp)/(total_tp + total_fn)

    # print(f"Avg precision : {avg_precision}")
    # print(f"Avg recall : {avg_recall}")

    json_obj = {"accuracy": correct/total, "precision": avg_precision,
                "recall": avg_recall, "total": total}
    return json_obj


def calculate_accuracy_metrics(original_grades, model_grades):
    '''
    Calculates different kinds of accuracy metrics

    Args : 
        original_grades (dict) : A dictionary of dictionary of dictionary of TA grades. Lab names are the first level keys. Criterion names are the second level keys. Student ids are the third level keys
        model_grades (dict) : A dictionary of dictionary of dictionary of model assigned grades. Lab names are the first level keys. Criterion names are the second level keys. Student ids are the third level keys

    Returns : 
        A tuple of : 
        criterion_wise_accuracies (dict) : A dictionary of criterion wise accuracies. Keys are criterion names 
        lab_wise_micro_accuracies (dict) : A dictionary of lab wise accuracies. Keys are lab names
        lab_wise_macro_accuracies (dict) : Criterion wise accuracies are first calculated and then averaged within a lab. Keys are lab names
        criterion_wise_macro_accuracy (float) : Accuracies are first calculated for all criteria and then averaged
        lab_wise_macro_accuracy (float) : Accuracies are first calculated for all labs and then averaged
        overall_accuracy (float) : The overall accuracy averaged over all submissions
    '''

    # Compute accuracy for all submissions for all ratings associated with a particular criterion
    criterion_wise_accuracies = {}
    # First calculate criterion wise accuracies and then average over all criteria accuracies within a lab
    lab_wise_macro_accuracies = {}
    # Compute accuracy for all submissions for all criteria and ratings for a particular lab
    lab_wise_micro_accuracies = {}
    overall_accuracy = 0  # Accuracy for all submissions for all labs, criteria, ratings

    total_sample_count = 0
    total_correct_count = 0

    criterion_wise_macro_accuracy = 0
    lab_wise_macro_accuracy = 0

    total_criteria = 0
    total_labs = 0

    for lab_name in model_grades.keys():
        total_labs += 1
        sample_count_within_lab = 0
        correct_count_within_lab = 0
        num_criteria_within_lab = 0

        lab_wise_macro_accuracies[lab_name] = 0
        lab_wise_micro_accuracies[lab_name] = 0
        criterion_wise_accuracies[lab_name] = {}

        for criterion_name in model_grades[lab_name]:
            sample_count_within_criterion = 0
            correct_count_within_criterion = 0

            criterion_wise_accuracies[lab_name][criterion_name] = {}

            for student_id in model_grades[lab_name][criterion_name].keys():
                if (student_id in original_grades[lab_name][criterion_name].keys()):
                    if (len(original_grades[lab_name][criterion_name][student_id]) != 1):
                        continue
                    sample_count_within_criterion += 1
                    if (original_grades[lab_name][criterion_name][student_id].strip() == model_grades[lab_name][criterion_name][student_id].strip()):
                        correct_count_within_criterion += 1

            sample_count_within_lab += sample_count_within_criterion
            correct_count_within_lab += correct_count_within_criterion
            num_criteria_within_lab += 1

            criterion_wise_accuracies[lab_name][criterion_name] = correct_count_within_criterion / \
                sample_count_within_criterion
            lab_wise_macro_accuracies[lab_name] += criterion_wise_accuracies[lab_name][criterion_name]

            criterion_wise_macro_accuracy += criterion_wise_accuracies[lab_name][criterion_name]

        total_sample_count += sample_count_within_lab
        total_correct_count += correct_count_within_lab
        total_criteria += num_criteria_within_lab

        lab_wise_macro_accuracies[lab_name] /= num_criteria_within_lab
        lab_wise_micro_accuracies[lab_name] = correct_count_within_lab / \
            sample_count_within_lab

        lab_wise_macro_accuracy += lab_wise_macro_accuracies[lab_name]

    overall_accuracy = total_correct_count/total_sample_count
    criterion_wise_macro_accuracy /= total_criteria
    lab_wise_macro_accuracy /= total_labs

    # for lab_name in model_grades.keys() :
    #     print("Lab name :", lab_name)
    #     print("Criterion wise accuracies : ")
    #     for criterion_name in model_grades[lab_name].keys() :
    #         print(f"{criterion_name} : {criterion_wise_accuracies[lab_name][criterion_name]}")
    #     print(f"Average accuracy for the lab : {lab_wise_micro_accuracies[lab_name]}\n\n")

    # print(f"Total number of samples : {total_sample_count}")
    # print(f"Accuracy averaged over criteria : {criterion_wise_macro_accuracy}")
    # print(f"Accuracy averaged over labs : {lab_wise_macro_accuracy}")
    # print(f"Overall accuracy : {overall_accuracy}")

    return criterion_wise_accuracies, lab_wise_micro_accuracies, lab_wise_macro_accuracies, criterion_wise_macro_accuracy, lab_wise_macro_accuracy, overall_accuracy


def calculate_precision_recall(original_grades, model_grades):
    '''
    Calculates different kinds of accuracy metrics

    Args : 
        original_grades (dict) : A dictionary of dictionary dictionary of TA grades. Lab names are the first level keys. Criterion names are the second level keys. Student ids are the third level keys
        model_grades (dict) : A dictionary of dictionary of dictionary model assigned grades. Lab names are the first level keys. Criterion names are the second level keys. Student ids are the third level keys

    Returns : 
        A tuple of : 
        tp, fp, tn, fn (dict) : Dictionary of dictionary of true positives, false positives, true negatives, false negatives. Lab names and criterion names are the keys at the two levels.
        rating_wise_metrics (dict) : Rating wise precision and recall 
        lab_wise_metrics : Lab wise precision and recall
        overall_rating_wise_metrics (dict) :  Dict of precision (float) and recall (float) averaged over all ratings 
        overall_criterion_wise_metrics (dict) : Dict of precision (float) and recall (float) averaged over all criteria
        overall_lab_wise_metrics (dict) : Dict of precision (float) and recall (float) averaged over all criteria
        original_rating_counts (dict) : Counts for each rating for each criterion for each lab assigned by the TAs
        model_rating_counts (dict) : Counts for each rating for each criterion for each lab assigned by the model
    '''
    rating_wise_metrics = {}  # Compute precision and recall for each rating
    # Compute precision and recall for each rating within a criterion. Then average these values within a criterion
    criterion_wise_metrics = {}
    lab_wise_metrics = {}  # Compute precision and recall for each rating within a criterion. Average these values within a criterion, then within a lab

    # Find average over all ratings
    overall_rating_wise_metrics = {"precision": 0, "recall": 0}
    # Find average over all criteria
    overall_criterion_wise_metrics = {"precision": 0, "recall": 0}
    overall_lab_wise_metrics = {"precision": 0,
                                "recall": 0}  # Find average over all labs

    tp, fp, tn, fn = {}, {}, {}, {}
    original_rating_counts = {}
    model_rating_counts = {}

    total_rating_count = 0
    total_criteria_count = 0
    total_lab_count = 0

    directory = "/raid/ganesh/nagakalyani/nagakalyani/ICER/dataset/"

    for lab_name in model_grades.keys():
        # print(lab_name)
        total_lab_count += 1
        tp[lab_name] = {}
        fp[lab_name] = {}
        tn[lab_name] = {}
        fn[lab_name] = {}

        original_rating_counts[lab_name] = {}
        model_rating_counts[lab_name] = {}

        rating_wise_metrics[lab_name] = {}
        criterion_wise_metrics[lab_name] = {}
        lab_wise_metrics[lab_name] = {"precision": 0, "recall": 0}

        rubrics = get_rubrics(os.path.join(directory, lab_name))

        lab_criteria_count = 0
        lab_rating_count = 0

        for criterion_name in model_grades[lab_name]:
            # print(criterion_name , ":", rubrics[criterion_name]["description"])
            lab_criteria_count += 1

            # For counting the no. of ratings in a particular criterion
            criterion_rating_count = 0

            criterion_wise_metrics[lab_name][criterion_name] = {
                "precision": 0, "recall": 0}
            rating_wise_metrics[lab_name][criterion_name] = {}

            tp[lab_name][criterion_name] = {}
            fp[lab_name][criterion_name] = {}
            tn[lab_name][criterion_name] = {}
            fn[lab_name][criterion_name] = {}

            original_rating_counts[lab_name][criterion_name] = {}
            model_rating_counts[lab_name][criterion_name] = {}

            # All possible ratings for a particular criterion of a lab
            for grade in rubrics[criterion_name]["ratings"].keys():
                grade = grade.strip().capitalize()
                tp[lab_name][criterion_name][grade] = 0
                fp[lab_name][criterion_name][grade] = 0
                tn[lab_name][criterion_name][grade] = 0
                fn[lab_name][criterion_name][grade] = 0

                original_rating_counts[lab_name][criterion_name][grade] = 0
                model_rating_counts[lab_name][criterion_name][grade] = 0

            possible_grades = tp[lab_name][criterion_name].keys()

            for student_id in model_grades[lab_name][criterion_name].keys():
                if (student_id in original_grades[lab_name][criterion_name].keys()):
                    original_grade = original_grades[lab_name][criterion_name][student_id].strip(
                    ).capitalize()
                    model_grade = model_grades[lab_name][criterion_name][student_id].strip(
                    ).capitalize()

                    original_rating_counts[lab_name][criterion_name][original_grade] += 1
                    model_rating_counts[lab_name][criterion_name][model_grade] += 1

                    if (original_grade == model_grade):
                        tp[lab_name][criterion_name][original_grade] += 1

                        for grade in possible_grades:
                            if (grade != original_grade):
                                tn[lab_name][criterion_name][grade] += 1
                    else:
                        fp[lab_name][criterion_name][model_grade] += 1
                        fn[lab_name][criterion_name][original_grade] += 1

                        for grade in possible_grades:
                            if ((grade != original_grade) and (grade != model_grade)):
                                tn[lab_name][criterion_name][grade] += 1

            # print("Original grades : ", end="")
            # for grade in sorted(possible_grades) :
            #     print(f"{grade} : {original_rating_counts[lab_name][criterion_name][grade]}", end=" ")
            # print("")

            # print("Model grades : ", end="")
            # for grade in sorted(possible_grades) :
            #     print(f"{grade} : {model_rating_counts[lab_name][criterion_name][grade]}", end=" ")
            # print("")

            for grade in sorted(possible_grades):
                criterion_rating_count += 1
                rating_wise_metrics[lab_name][criterion_name][grade] = {}

                # First obtain the tp, fp, tn, fn for that particular rating
                grade_tp = tp[lab_name][criterion_name][grade]
                grade_fp = fp[lab_name][criterion_name][grade]
                grade_tn = tn[lab_name][criterion_name][grade]
                grade_fn = fn[lab_name][criterion_name][grade]

                # Compute metrics for each rating
                if ((grade_tp + grade_fp > 0)):
                    rating_wise_metrics[lab_name][criterion_name][grade]["precision"] = grade_tp/(
                        grade_tp + grade_fp)
                elif (grade_fn > 0):
                    rating_wise_metrics[lab_name][criterion_name][grade]["precision"] = 0
                else:
                    rating_wise_metrics[lab_name][criterion_name][grade]["precision"] = 1

                if ((grade_tp + grade_fn > 0)):
                    rating_wise_metrics[lab_name][criterion_name][grade]["recall"] = grade_tp/(
                        grade_tp + grade_fn)
                else:
                    rating_wise_metrics[lab_name][criterion_name][grade]["recall"] = 1

                rating_precision = rating_wise_metrics[lab_name][criterion_name][grade]["precision"]
                rating_recall = rating_wise_metrics[lab_name][criterion_name][grade]["recall"]

                # Aggregate them criterion wise
                criterion_wise_metrics[lab_name][criterion_name]["precision"] += rating_precision
                criterion_wise_metrics[lab_name][criterion_name]["recall"] += rating_recall

                # Aggregate them lab wise
                lab_wise_metrics[lab_name]["precision"] += rating_precision
                lab_wise_metrics[lab_name]["recall"] += rating_recall

                # Aggregate them for overall metrics
                overall_rating_wise_metrics["precision"] += rating_precision
                overall_rating_wise_metrics["recall"] += rating_recall

                # print(f"{grade} - TP : {grade_tp}, FP : {grade_fp}, TN : {grade_tn}, FN : {grade_fn}, Precision : {rating_precision}, Recall : {rating_recall}")

            # Calculate metrics for each criterion
            criterion_wise_metrics[lab_name][criterion_name]["precision"] /= criterion_rating_count
            criterion_wise_metrics[lab_name][criterion_name]["recall"] /= criterion_rating_count

            criterion_precision = criterion_wise_metrics[lab_name][criterion_name]["precision"]
            criterion_recall = criterion_wise_metrics[lab_name][criterion_name]["recall"]

            # print(f"Criterion precision : {criterion_precision} Criterion recall : {criterion_recall}\n")

            overall_criterion_wise_metrics["precision"] += criterion_precision
            overall_criterion_wise_metrics["recall"] += criterion_recall

            lab_rating_count += criterion_rating_count

        total_rating_count += lab_rating_count
        total_criteria_count += lab_criteria_count

        lab_wise_metrics[lab_name]["precision"] /= lab_rating_count
        lab_wise_metrics[lab_name]["recall"] /= lab_rating_count

        overall_lab_wise_metrics["precision"] += lab_wise_metrics[lab_name]["precision"]
        overall_lab_wise_metrics["recall"] += lab_wise_metrics[lab_name]["recall"]

        # print("\n\n")

    # Calculate overall metric values
    overall_rating_wise_metrics["precision"] /= total_rating_count
    overall_rating_wise_metrics["recall"] /= total_rating_count

    overall_criterion_wise_metrics["precision"] /= total_criteria_count
    overall_criterion_wise_metrics["recall"] /= total_criteria_count

    overall_lab_wise_metrics["precision"] /= total_lab_count
    overall_lab_wise_metrics["recall"] /= total_lab_count

    return tp, fp, tn, fn, rating_wise_metrics, criterion_wise_metrics, lab_wise_metrics, overall_rating_wise_metrics, overall_criterion_wise_metrics, overall_lab_wise_metrics, original_rating_counts, model_rating_counts


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


def compare_zs_fs(zs_folder_name, fs_folder_name, results_path):
    '''
    Compares the performance of 0-shot prompting and few-shot prompting across various metrics 

    Args : 
        zs_folder_name : The folder where 0-shot results are stored
        fs_folder_name : The folder where few-shot results are stored
        results_path   : The path of the file where the comparison results are dumped

    Returns : 
        None
    '''
    directory = "/raid/ganesh/nagakalyani/nagakalyani/ICER/dataset/"
    results_par_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/"

    zs_path = os.path.join(results_par_folder, zs_folder_name)
    fs_path = os.path.join(results_par_folder, fs_folder_name)

    results = {}

    for lab in os.listdir(directory):
        if lab.startswith("CP"):
            print(lab)
            results[lab] = {}
            lab_path = os.path.join(directory, lab)
            rubrics = get_rubrics(lab_path)

            for criterion in rubrics.keys():
                print(criterion, ":", rubrics[criterion]["description"])
                results[lab][criterion] = {}
                original_grades = extract_original_criterion_grade(
                    lab_path, criterion)
                zs_grades = extract_llm_ratings(
                    os.path.join(zs_path, lab), criterion)
                fs_grades = extract_llm_ratings(
                    os.path.join(fs_path, lab), criterion)

                original_option_counts = Counter(original_grades.values())
                zs_option_counts = Counter(zs_grades.values())
                fs_option_counts = Counter(fs_grades.values())

                # Display the counts
                print("Chosen Option Statistics : ")
                print("Original TA Grades")
                for option, count in sorted(original_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("Zero shot")
                for option, count in sorted(zs_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("Few shot")
                for option, count in sorted(fs_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("Results : ")
                print("Zero shot")
                results[lab][criterion]["zero_shot"] = calculate_metrics(
                    original_grades, zs_grades)

                print("Few shot")
                results[lab][criterion]["few_shot"] = calculate_metrics(
                    original_grades, fs_grades)

    with open(results_path, "w") as f:
        json.dump(results, f)


def compare_all(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, results_path=None):
    '''
    Compares the performance of prompting and fine-tuning methods across various metrics 

    Args : 
        zs_folder_name : The folder where 0-shot results are stored
        fs_folder_name : The folder where few-shot results are stored
        results_path   : The path of the file where the comparison results are dumped

    Returns : 
        None
    '''
    directory = "/raid/ganesh/nagakalyani/nagakalyani/ICER/dataset/"
    results_par_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/"

    zs_path = os.path.join(results_par_folder, zs_folder_name)
    fs_path = os.path.join(results_par_folder, fs_folder_name)
    sft_path = os.path.join(results_par_folder, sft_folder_name)
    dpo_path = os.path.join(results_par_folder, dpo_folder_name)

    results = {}

    for lab in os.listdir(directory):
        if lab.startswith("CP"):
            print(lab)
            results[lab] = {}
            lab_path = os.path.join(directory, lab)
            rubrics = get_rubrics(lab_path)

            for criterion in rubrics.keys():
                print(criterion, ":", rubrics[criterion]["description"])
                results[lab][criterion] = {}
                original_grades = extract_original_criterion_grade(
                    lab_path, criterion)
                zs_grades = extract_llm_ratings(
                    os.path.join(zs_path, lab), criterion)
                fs_grades = extract_llm_ratings(
                    os.path.join(fs_path, lab), criterion)
                sft_grades = extract_llm_ratings(
                    os.path.join(sft_path, lab), criterion)
                dpo_grades = extract_llm_ratings(
                    os.path.join(dpo_path, lab), criterion)

                zsr = {}
                fsr = {}
                sftr = {}
                original_reduced = {}

                for key in dpo_grades.keys():
                    if (key in zs_grades.keys()):
                        zsr[key] = zs_grades[key]
                    if (key in fs_grades.keys()):
                        fsr[key] = fs_grades[key]
                    if (key in sft_grades.keys()):
                        sftr[key] = sft_grades[key]
                    if (key in original_grades.keys()):
                        original_reduced[key] = original_grades[key]

                zs_grades = zsr
                fs_grades = fsr
                sft_grades = sftr

                original_option_counts = Counter(original_reduced.values())
                zs_option_counts = Counter(zs_grades.values())
                fs_option_counts = Counter(fs_grades.values())
                sft_option_counts = Counter(sft_grades.values())
                dpo_option_counts = Counter(dpo_grades.values())

                most_freq_option = 'A'
                max_freq = 0

                for option, count in sorted(original_option_counts.items()):
                    if (count > max_freq):
                        most_freq_option = option
                        max_freq = count

                # 0R grades
                zero_r_grades = {}
                for key in zs_grades.keys():
                    zero_r_grades[key] = most_freq_option

                # Display the counts
                print("Chosen Option Statistics : ")
                print("Original TA Grades")
                for option, count in sorted(original_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("Zero shot")
                for option, count in sorted(zs_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("Few shot")
                for option, count in sorted(fs_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("LoRA")
                for option, count in sorted(sft_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                print("DPO")
                for option, count in sorted(dpo_option_counts.items()):
                    print(f"{option}: {count}", end=" ")
                print("\n", end="")

                # Display the results
                print("Results : ")
                print("Zero shot")
                results[lab][criterion]["zero_shot"] = calculate_metrics(
                    original_grades, zs_grades)

                print("Few shot")
                results[lab][criterion]["few_shot"] = calculate_metrics(
                    original_grades, fs_grades)

                print("LoRA")
                results[lab][criterion]["sft"] = calculate_metrics(
                    original_grades, sft_grades)

                print("DPO")
                results[lab][criterion]["dpo"] = calculate_metrics(
                    original_grades, dpo_grades)

                print("0R")
                results[lab][criterion]["zr"] = calculate_metrics(
                    original_grades, zero_r_grades)
                print("\n\n")

    if (results_path):
        with open(results_path, "w") as f:
            json.dump(results, f)


def analyse_zs_fs_results(results_path, common_criteria=0, uncommon_criteria=0, eval_labs=None, use_eval=1, save_plot=1, plot_name=None):
    with open(results_path, "r") as f:
        content = json.load(f)

    labs_to_consider = []
    if (eval_labs and use_eval):
        labs_to_consider = eval_labs
    elif (eval_labs):
        for lab in content.keys():
            if (lab not in eval_labs):
                labs_to_consider.append(lab)
    else:
        labs_to_consider = content.keys()

    lab_wise_results = {}
    zsa, fsa = 0, 0  # Accuracy
    zsp, fsp = 0, 0  # Precision
    zsr, fsr = 0, 0  # Recall

    zs_criterion_count = 0
    fs_criterion_count = 0

    for lab in labs_to_consider:
        lab_criteria_zs_count = 0
        lab_criteria_fs_count = 0

        lab_wise_results[lab] = {
            "zero_shot": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0
            },
            "few_shot": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0
            }
        }

        criteria_to_consider = []
        if (uncommon_criteria):
            for criterion in content[lab].keys():
                if (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                    criteria_to_consider.append(criterion)
        elif (common_criteria):
            for criterion in content[lab].keys():
                if not (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                    criteria_to_consider.append(criterion)
        else:
            criteria_to_consider = content[lab].keys()

        for criterion in criteria_to_consider:
            lab_criteria_zs_count += 1
            zs_results = content[lab][criterion]["zero_shot"]
            fs_results = content[lab][criterion]["few_shot"]

            zs_criterion_count += 1
            zsa += zs_results["accuracy"]
            zsp += zs_results["precision"]
            zsr += zs_results["recall"]

            lab_wise_results[lab]["zero_shot"]["accuracy"] += zs_results["accuracy"]
            lab_wise_results[lab]["zero_shot"]["precision"] += zs_results["precision"]
            lab_wise_results[lab]["zero_shot"]["recall"] += zs_results["recall"]

            if (content[lab][criterion]["few_shot"]):
                lab_criteria_fs_count += 1
                fs_criterion_count += 1
                fsa += fs_results["accuracy"]
                fsp += fs_results["precision"]
                fsr += fs_results["recall"]

                lab_wise_results[lab]["few_shot"]["accuracy"] += fs_results["accuracy"]
                lab_wise_results[lab]["few_shot"]["precision"] += fs_results["precision"]
                lab_wise_results[lab]["few_shot"]["recall"] += fs_results["recall"]

        lab_wise_results[lab]["zero_shot"]["accuracy"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["precision"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["recall"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["count"] = lab_criteria_zs_count

        lab_wise_results[lab]["few_shot"]["accuracy"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["precision"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["recall"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["count"] = lab_criteria_fs_count

    zsa /= zs_criterion_count
    zsp /= zs_criterion_count
    zsr /= zs_criterion_count

    fsa /= fs_criterion_count
    fsp /= fs_criterion_count
    fsr /= fs_criterion_count

    print("Overall comparison :")
    print(f"Criteria count : {zs_criterion_count} {fs_criterion_count}")
    print(f"Accuracy : {zsa} {fsa}")
    print(f"Precision : {zsp} {fsp}")
    print(f"Recall : {zsr} {fsr}")

    print("\n\n")

    print("Lab wise comparison : ", end="")
    if (uncommon_criteria):
        print("(Unique criteria)", end=" ")
    if (eval_labs):
        print("(Only eval labs)")
    else:
        print("")
    for lab in lab_wise_results.keys():
        print(lab)
        print("Criteria count :", lab_wise_results[lab]["zero_shot"]
              ["count"], lab_wise_results[lab]["few_shot"]["count"])
        print("Accuracy :", lab_wise_results[lab]["zero_shot"]
              ["accuracy"], lab_wise_results[lab]["few_shot"]["accuracy"])
        print("Precision :", lab_wise_results[lab]["zero_shot"]
              ["precision"], lab_wise_results[lab]["few_shot"]["precision"])
        print("Recall :", lab_wise_results[lab]["zero_shot"]
              ["recall"], lab_wise_results[lab]["few_shot"]["recall"])

    if save_plot:
        sorted_labs = sorted(lab_wise_results.keys())

        metrics = ["accuracy", "precision", "recall"]
        for metric in metrics:
            zs_metrics = [lab_wise_results[lab]["zero_shot"][metric]
                          for lab in sorted_labs]
            fs_metrics = [lab_wise_results[lab]["few_shot"][metric]
                          for lab in sorted_labs]

            bar_width = 0.125
            x = np.arange(len(sorted_labs))

            plt.bar(x - bar_width, zs_metrics, width=2 *
                    bar_width, label='Zero-shot')
            plt.bar(x + bar_width, fs_metrics,
                    width=2*bar_width, label='Few-shot')

            plt.xlabel('Lab')
            plt.ylabel(metric.capitalize())
            plt.title(f'Comparison of zero-shot and few-shot {metric}')
            plt.xticks(x, np.arange(len(sorted_labs)))
            plt.legend()

            # Save the plot
            plt.savefig(f"Plots/{metric.capitalize()}/{plot_name}.png")
            plt.clf()


def analyse_all_results(results_path, common_criteria=0, uncommon_criteria=0, eval_labs=None, use_eval=1, save_plot=1, plot_name=None):
    with open(results_path, "r") as f:
        content = json.load(f)

    labs_to_consider = []
    if (eval_labs and use_eval):
        labs_to_consider = eval_labs
    elif (eval_labs):
        for lab in content.keys():
            if (lab not in eval_labs):
                labs_to_consider.append(lab)
    else:
        labs_to_consider = content.keys()

    lab_wise_results = {}
    zsa, fsa, sfta, dpoa, zra = 0, 0, 0, 0, 0  # Accuracy
    zsp, fsp, sftp, dpop, zrp = 0, 0, 0, 0, 0  # Precision
    zsr, fsr, sftr, dpor, zrr = 0, 0, 0, 0, 0  # Recall

    zs_criterion_count, fs_criterion_count, sft_criterion_count, dpo_criterion_count, zr_criterion_count = 0, 0, 0, 0, 0
    zs_total, fs_total, sft_total, dpo_total, zr_total = 0, 0, 0, 0, 0

    for lab in labs_to_consider:
        lab_criteria_zs_count, lab_criteria_fs_count, lab_criteria_sft_count, lab_criteria_dpo_count, lab_criteria_zr_count = 0, 0, 0, 0, 0

        lab_wise_results[lab] = {
            "zero_shot": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0,
                "total": 0
            },
            "few_shot": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0,
                "total": 0
            },
            "sft": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0,
                "total": 0
            },
            "dpo": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0,
                "total": 0
            },
            "zr": {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "count": 0,
                "total": 0
            }
        }

        criteria_to_consider = []
        if (uncommon_criteria):
            for criterion in content[lab].keys():
                if (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                    criteria_to_consider.append(criterion)
        elif (common_criteria):
            for criterion in content[lab].keys():
                if not (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                    criteria_to_consider.append(criterion)
        else:
            criteria_to_consider = content[lab].keys()

        for criterion in criteria_to_consider:
            lab_criteria_zs_count += 1
            lab_criteria_sft_count += 1
            lab_criteria_dpo_count += 1
            lab_criteria_zr_count += 1

            zs_results = content[lab][criterion]["zero_shot"]
            fs_results = content[lab][criterion]["few_shot"]
            sft_results = content[lab][criterion]["sft"]
            dpo_results = content[lab][criterion]["dpo"]
            zr_results = content[lab][criterion]["zr"]

            zs_total += zs_results["total"]
            fs_total += fs_results["total"]
            sft_total += sft_results["total"]
            dpo_total += dpo_results["total"]
            zr_total += zr_results["total"]

            zs_criterion_count += 1
            zsa += zs_results["accuracy"]
            zsp += zs_results["precision"]
            zsr += zs_results["recall"]

            lab_wise_results[lab]["zero_shot"]["accuracy"] += zs_results["accuracy"]
            lab_wise_results[lab]["zero_shot"]["precision"] += zs_results["precision"]
            lab_wise_results[lab]["zero_shot"]["recall"] += zs_results["recall"]

            if (content[lab][criterion]["few_shot"]):
                lab_criteria_fs_count += 1
                fs_criterion_count += 1
                fsa += fs_results["accuracy"]
                fsp += fs_results["precision"]
                fsr += fs_results["recall"]

                lab_wise_results[lab]["few_shot"]["accuracy"] += fs_results["accuracy"]
                lab_wise_results[lab]["few_shot"]["precision"] += fs_results["precision"]
                lab_wise_results[lab]["few_shot"]["recall"] += fs_results["recall"]

            sft_criterion_count += 1
            sfta += sft_results["accuracy"]
            sftp += sft_results["precision"]
            sftr += sft_results["recall"]

            lab_wise_results[lab]["sft"]["accuracy"] += sft_results["accuracy"]
            lab_wise_results[lab]["sft"]["precision"] += sft_results["precision"]
            lab_wise_results[lab]["sft"]["recall"] += sft_results["recall"]

            dpo_criterion_count += 1
            dpoa += dpo_results["accuracy"]
            dpop += dpo_results["precision"]
            dpor += dpo_results["recall"]

            lab_wise_results[lab]["dpo"]["accuracy"] += dpo_results["accuracy"]
            lab_wise_results[lab]["dpo"]["precision"] += dpo_results["precision"]
            lab_wise_results[lab]["dpo"]["recall"] += dpo_results["recall"]

            zr_criterion_count += 1
            zra += zr_results["accuracy"]
            zrp += zr_results["precision"]
            zrr += zr_results["recall"]

            lab_wise_results[lab]["zr"]["accuracy"] += zr_results["accuracy"]
            lab_wise_results[lab]["zr"]["precision"] += zr_results["precision"]
            lab_wise_results[lab]["zr"]["recall"] += zr_results["recall"]

        lab_wise_results[lab]["zero_shot"]["accuracy"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["precision"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["recall"] /= lab_criteria_zs_count
        lab_wise_results[lab]["zero_shot"]["count"] = lab_criteria_zs_count

        lab_wise_results[lab]["few_shot"]["accuracy"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["precision"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["recall"] /= lab_criteria_fs_count
        lab_wise_results[lab]["few_shot"]["count"] = lab_criteria_fs_count

        lab_wise_results[lab]["sft"]["accuracy"] /= lab_criteria_sft_count
        lab_wise_results[lab]["sft"]["precision"] /= lab_criteria_sft_count
        lab_wise_results[lab]["sft"]["recall"] /= lab_criteria_sft_count
        lab_wise_results[lab]["sft"]["count"] = lab_criteria_sft_count

        lab_wise_results[lab]["dpo"]["accuracy"] /= lab_criteria_dpo_count
        lab_wise_results[lab]["dpo"]["precision"] /= lab_criteria_dpo_count
        lab_wise_results[lab]["dpo"]["recall"] /= lab_criteria_dpo_count
        lab_wise_results[lab]["dpo"]["count"] = lab_criteria_dpo_count

        lab_wise_results[lab]["zr"]["accuracy"] /= lab_criteria_zr_count
        lab_wise_results[lab]["zr"]["precision"] /= lab_criteria_zr_count
        lab_wise_results[lab]["zr"]["recall"] /= lab_criteria_zr_count
        lab_wise_results[lab]["zr"]["count"] = lab_criteria_zr_count

    zsa /= zs_criterion_count
    zsp /= zs_criterion_count
    zsr /= zs_criterion_count

    fsa /= fs_criterion_count
    fsp /= fs_criterion_count
    fsr /= fs_criterion_count

    sfta /= sft_criterion_count
    sftp /= sft_criterion_count
    sftr /= sft_criterion_count

    dpoa /= dpo_criterion_count
    dpop /= dpo_criterion_count
    dpor /= dpo_criterion_count

    zra /= zr_criterion_count
    zrp /= zr_criterion_count
    zrr /= zr_criterion_count

    print("Overall comparison :")
    print(
        f"Criteria count : {zs_criterion_count} {fs_criterion_count} {sft_criterion_count} {dpo_criterion_count} {zr_criterion_count}")
    print(f"Accuracy : {zsa} {fsa} {sfta} {dpoa} {zra}")
    print(f"Precision : {zsp} {fsp} {sftp} {dpop} {zrp}")
    print(f"Recall : {zsr} {fsr} {sftr} {dpor} {zrr}")
    print(f"Total : {zs_total} {fs_total} {sft_total} {dpo_total} {zr_total}")

    print("\n\n")

    print("Lab wise comparison : ", end="")
    if (uncommon_criteria):
        print("(Unique criteria)", end=" ")
    if (common_criteria):
        print("(Common criteria)", end=" ")
    if (eval_labs):
        print("(Only eval labs)")
    else:
        print("")
    for lab in lab_wise_results.keys():
        print(lab)
        print("Criteria count :", lab_wise_results[lab]["zero_shot"]["count"], lab_wise_results[lab]["few_shot"]["count"],
              lab_wise_results[lab]["sft"]["count"], lab_wise_results[lab]["dpo"]["count"], lab_wise_results[lab]["zr"]["count"])
        print("Accuracy :", lab_wise_results[lab]["zero_shot"]["accuracy"], lab_wise_results[lab]["few_shot"]["accuracy"],
              lab_wise_results[lab]["sft"]["accuracy"], lab_wise_results[lab]["dpo"]["accuracy"], lab_wise_results[lab]["zr"]["accuracy"])
        print("Precision :", lab_wise_results[lab]["zero_shot"]["precision"], lab_wise_results[lab]["few_shot"]["precision"],
              lab_wise_results[lab]["sft"]["precision"], lab_wise_results[lab]["dpo"]["precision"], lab_wise_results[lab]["zr"]["precision"])
        print("Recall :", lab_wise_results[lab]["zero_shot"]["recall"], lab_wise_results[lab]["few_shot"]["recall"],
              lab_wise_results[lab]["sft"]["recall"], lab_wise_results[lab]["dpo"]["recall"], lab_wise_results[lab]["zr"]["recall"])

    if save_plot:
        sorted_labs = sorted(lab_wise_results.keys())

        metrics = ["accuracy", "precision", "recall"]
        for metric in metrics:
            zs_accuracies = [lab_wise_results[lab]
                             ["zero_shot"][metric] for lab in sorted_labs]
            fs_accuracies = [lab_wise_results[lab]
                             ["few_shot"][metric] for lab in sorted_labs]
            sft_accuracies = [lab_wise_results[lab]["sft"][metric]
                              for lab in sorted_labs]
            dpo_accuracies = [lab_wise_results[lab]["dpo"][metric]
                              for lab in sorted_labs]

            bar_width = 0.0625
            x = np.arange(len(sorted_labs))

            plt.bar(x - 3*bar_width, zs_accuracies,
                    width=2*bar_width, label='Zero-shot')
            plt.bar(x - bar_width, fs_accuracies,
                    width=2*bar_width, label='Zero-shot')
            plt.bar(x + bar_width, sft_accuracies,
                    width=2*bar_width, label='SFT')
            plt.bar(x + 3*bar_width, dpo_accuracies,
                    width=2*bar_width, label='DPO')

            plt.xlabel('Lab')
            plt.ylabel(f'{metric.capitalize()}')
            plt.title(f'{metric.capitalize()} comparison from various methods')
            plt.xticks(x, np.arange(len(sorted_labs)))
            plt.legend()

            # Save the plot
            plt.savefig(f"Plots/{metric.capitalize()}/{plot_name}.png")
            plt.clf()


def analyse_results(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, common_criteria=0, uncommon_criteria=0, eval_labs=None, use_eval=0, save_path=None, all_zs_fs=False, zr_only_train=0, save_comparison_path=None):
    directory = "/raid/ganesh/nagakalyani/nagakalyani/ICER/dataset/"
    results_par_folder = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/"

    original_grades = {}
    zs_grades = {}
    fs_grades = {}
    sft_grades = {}
    dpo_grades = {}
    zr_grades = {}

    zs_path = os.path.join(results_par_folder, zs_folder_name)
    fs_path = os.path.join(results_par_folder, fs_folder_name)
    sft_path = os.path.join(results_par_folder, sft_folder_name)
    dpo_path = os.path.join(results_par_folder, dpo_folder_name)

    labs_to_consider = []
    if (eval_labs and use_eval):
        labs_to_consider = eval_labs
    elif (eval_labs):
        for lab in os.listdir(directory):
            if (lab not in eval_labs) and (lab.startswith("CP")):
                labs_to_consider.append(lab)
    else:
        for lab in os.listdir(directory):
            if (lab.startswith("CP")):
                labs_to_consider.append(lab)

    # Keep track of what all criteria we're using
    criteria_used = {}

    for lab in labs_to_consider:
        if lab.startswith("CP"):
            original_grades[lab] = {}
            zs_grades[lab] = {}
            fs_grades[lab] = {}
            sft_grades[lab] = {}
            dpo_grades[lab] = {}
            zr_grades[lab] = {}

            lab_path = os.path.join(directory, lab)
            rubrics = get_rubrics(lab_path)

            criteria_to_consider = []
            if (uncommon_criteria):
                for criterion in rubrics.keys():
                    if (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                        criteria_to_consider.append(criterion)
            elif (common_criteria):
                for criterion in rubrics.keys():
                    if not (criterion.startswith("Criterion") or criterion.startswith("criterion") or criterion.startswith("Criteria")):
                        criteria_to_consider.append(criterion)
            else:
                criteria_to_consider = rubrics.keys()

            criteria_used[lab] = criteria_to_consider

            for criterion in criteria_to_consider:
                original_grds = extract_original_criterion_grade(
                    lab_path, criterion)
                zs_grds = extract_llm_ratings(
                    os.path.join(zs_path, lab), criterion)
                fs_grds = extract_llm_ratings(
                    os.path.join(fs_path, lab), criterion)
                sft_grds = extract_llm_ratings(
                    os.path.join(sft_path, lab), criterion)
                dpo_grds = extract_llm_ratings(
                    os.path.join(dpo_path, lab), criterion)

                zr_grds = {}

                zsr = {}
                fsr = {}
                sftr = {}
                original_red = {}

                for key in dpo_grds.keys():
                    if not (all_zs_fs):
                        if (key in zs_grds.keys()):
                            zsr[key] = zs_grds[key]
                        if (key in fs_grds.keys()):
                            fsr[key] = fs_grds[key]
                        if (key in original_grds.keys()):
                            original_red[key] = original_grds[key]
                    if (key in sft_grds.keys()):
                        sftr[key] = sft_grds[key]

                if not (all_zs_fs):
                    zs_grds = zsr
                    fs_grds = fsr
                    original_grds = original_red
                sft_grds = sftr

                original_opt_counts = Counter(original_grds.values())

                if (zr_only_train):
                    gt_grds = {}
                    for key in original_grds.keys():
                        if (key not in dpo_grades.keys()):
                            gt_grds[key] = original_grds[key]
                    gt_opt_counts = Counter(gt_grds.values())
                else:
                    gt_opt_counts = original_opt_counts

                max_freq_option = 'A'
                curr_max_freq = 0
                for key, value in gt_opt_counts.items():
                    if (value > curr_max_freq):
                        max_freq_option = key
                        curr_max_freq = value

                for key in dpo_grds.keys():
                    zr_grds[key] = max_freq_option

                original_grades[lab][criterion] = original_grds
                zs_grades[lab][criterion] = zs_grds
                fs_grades[lab][criterion] = fs_grds
                sft_grades[lab][criterion] = sft_grds
                dpo_grades[lab][criterion] = dpo_grds
                zr_grades[lab][criterion] = zr_grds

    # Accuracy metrics
    zs_criterion_wise_accuracies, zs_lab_wise_micro_accuracies, zs_lab_wise_macro_accuracies, zs_criterion_wise_macro_accuracy, zs_lab_wise_macro_accuracy, zs_overall_accuracy = calculate_accuracy_metrics(
        original_grades, zs_grades)
    fs_criterion_wise_accuracies, fs_lab_wise_micro_accuracies, fs_lab_wise_macro_accuracies, fs_criterion_wise_macro_accuracy, fs_lab_wise_macro_accuracy, fs_overall_accuracy = calculate_accuracy_metrics(
        original_grades, fs_grades)
    sft_criterion_wise_accuracies, sft_lab_wise_micro_accuracies, sft_lab_wise_macro_accuracies, sft_criterion_wise_macro_accuracy, sft_lab_wise_macro_accuracy, sft_overall_accuracy = calculate_accuracy_metrics(
        original_grades, sft_grades)
    dpo_criterion_wise_accuracies, dpo_lab_wise_micro_accuracies, dpo_lab_wise_macro_accuracies, dpo_criterion_wise_macro_accuracy, dpo_lab_wise_macro_accuracy, dpo_overall_accuracy = calculate_accuracy_metrics(
        original_grades, dpo_grades)
    zr_criterion_wise_accuracies, zr_lab_wise_micro_accuracies, zr_lab_wise_macro_accuracies, zr_criterion_wise_macro_accuracy, zr_lab_wise_macro_accuracy, zr_overall_accuracy = calculate_accuracy_metrics(
        original_grades, zr_grades)

    total_criteria_used = 0
    for lab in criteria_used:
        total_criteria_used += len(criteria_used[lab])

    print("Total number of criteria :", total_criteria_used)

    print("Accuracy")
    print("Averaged over criteria : {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(zs_criterion_wise_macro_accuracy,
          fs_criterion_wise_macro_accuracy, sft_criterion_wise_macro_accuracy, dpo_criterion_wise_macro_accuracy, zr_criterion_wise_macro_accuracy))
    print("Averaged over labs     : {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        zs_lab_wise_macro_accuracy, fs_lab_wise_macro_accuracy, sft_lab_wise_macro_accuracy, dpo_lab_wise_macro_accuracy, zr_lab_wise_macro_accuracy))
    print("Overall accuracy       : {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        zs_overall_accuracy, fs_overall_accuracy, sft_overall_accuracy, dpo_overall_accuracy, zr_overall_accuracy))

    print("\n")

    # Precision and recall metrics
    zs_tp, zs_fp, zs_tn, zs_fn, zs_rating_wise_metrics, zs_criterion_wise_metrics, zs_lab_wise_metrics, zs_overall_rating_wise_metrics, zs_overall_criterion_wise_metrics, zs_overall_lab_wise_metrics, zs_original_rating_counts, zs_model_rating_counts = calculate_precision_recall(
        original_grades, zs_grades)
    fs_tp, fs_fp, fs_tn, fs_fn, fs_rating_wise_metrics, fs_criterion_wise_metrics, fs_lab_wise_metrics, fs_overall_rating_wise_metrics, fs_overall_criterion_wise_metrics, fs_overall_lab_wise_metrics, fs_original_rating_counts, fs_model_rating_counts = calculate_precision_recall(
        original_grades, fs_grades)
    sft_tp, sft_fp, sft_tn, sft_fn, sft_rating_wise_metrics, sft_criterion_wise_metrics, sft_lab_wise_metrics, sft_overall_rating_wise_metrics, sft_overall_criterion_wise_metrics, sft_overall_lab_wise_metrics, sft_original_rating_counts, sft_model_rating_counts = calculate_precision_recall(
        original_grades, sft_grades)
    dpo_tp, dpo_fp, dpo_tn, dpo_fn, dpo_rating_wise_metrics, dpo_criterion_wise_metrics, dpo_lab_wise_metrics, dpo_overall_rating_wise_metrics, dpo_overall_criterion_wise_metrics, dpo_overall_lab_wise_metrics, dpo_original_rating_counts, dpo_model_rating_counts = calculate_precision_recall(
        original_grades, dpo_grades)
    zr_tp, zr_fp, zr_tn, zr_fn, zr_rating_wise_metrics, zr_criterion_wise_metrics, zr_lab_wise_metrics, zr_overall_rating_wise_metrics, zr_overall_criterion_wise_metrics, zr_overall_lab_wise_metrics, zr_original_rating_counts, zr_model_rating_counts = calculate_precision_recall(
        original_grades, zr_grades)

    print("Precision and recall")

    print("Averaged over ratings :")
    print("Zero shot : {:.2f} {:.2f}".format(
        zs_overall_rating_wise_metrics["precision"], zs_overall_rating_wise_metrics["recall"]))
    print("Few shot  : {:.2f} {:.2f}".format(
        fs_overall_rating_wise_metrics["precision"], fs_overall_rating_wise_metrics["recall"]))
    print("SFT       : {:.2f} {:.2f}".format(
        sft_overall_rating_wise_metrics["precision"], sft_overall_rating_wise_metrics["recall"]))
    print("DPO       : {:.2f} {:.2f}".format(
        dpo_overall_rating_wise_metrics["precision"], dpo_overall_rating_wise_metrics["recall"]))
    print("0R        : {:.2f} {:.2f}".format(
        zr_overall_rating_wise_metrics["precision"], zr_overall_rating_wise_metrics["recall"]))

    print("Averaged over criteria : ")
    print("Zero shot : {:.2f} {:.2f}".format(
        zs_overall_criterion_wise_metrics["precision"], zs_overall_criterion_wise_metrics["recall"]))
    print("Few shot  : {:.2f} {:.2f}".format(
        fs_overall_criterion_wise_metrics["precision"], fs_overall_criterion_wise_metrics["recall"]))
    print("SFT       : {:.2f} {:.2f}".format(
        sft_overall_criterion_wise_metrics["precision"], sft_overall_criterion_wise_metrics["recall"]))
    print("DPO       : {:.2f} {:.2f}".format(
        dpo_overall_criterion_wise_metrics["precision"], dpo_overall_criterion_wise_metrics["recall"]))
    print("0R        : {:.2f} {:.2f}".format(
        zr_overall_criterion_wise_metrics["precision"], zr_overall_criterion_wise_metrics["recall"]))

    print("Averaged over labs :")
    print("Zero shot : {:.2f} {:.2f}".format(
        zs_overall_lab_wise_metrics["precision"], zs_overall_lab_wise_metrics["recall"]))
    print("Few shot  : {:.2f} {:.2f}".format(
        fs_overall_lab_wise_metrics["precision"], fs_overall_lab_wise_metrics["recall"]))
    print("SFT       : {:.2f} {:.2f}".format(
        sft_overall_lab_wise_metrics["precision"], sft_overall_lab_wise_metrics["recall"]))
    print("DPO       : {:.2f} {:.2f}".format(
        dpo_overall_lab_wise_metrics["precision"], dpo_overall_lab_wise_metrics["recall"]))
    print("0R        : {:.2f} {:.2f}".format(
        zr_overall_lab_wise_metrics["precision"], zr_overall_lab_wise_metrics["recall"]))

    print("\n")

    for lab in labs_to_consider:
        print(lab)
        print("Zero shot : {:.2f} {:.2f} {:.2f}".format(
            zs_lab_wise_macro_accuracies[lab], zs_lab_wise_metrics[lab]["precision"], zs_lab_wise_metrics[lab]["recall"]))
        print("Few shot  : {:.2f} {:.2f} {:.2f}".format(
            fs_lab_wise_macro_accuracies[lab], fs_lab_wise_metrics[lab]["precision"], fs_lab_wise_metrics[lab]["recall"]))
        print("SFT       : {:.2f} {:.2f} {:.2f}".format(
            sft_lab_wise_macro_accuracies[lab], sft_lab_wise_metrics[lab]["precision"], sft_lab_wise_metrics[lab]["recall"]))
        print("DPO       : {:.2f} {:.2f} {:.2f}".format(
            dpo_lab_wise_macro_accuracies[lab], dpo_lab_wise_metrics[lab]["precision"], dpo_lab_wise_metrics[lab]["recall"]))
        print("0R        : {:.2f} {:.2f} {:.2f}".format(
            zr_lab_wise_macro_accuracies[lab], zr_lab_wise_metrics[lab]["precision"], zr_lab_wise_metrics[lab]["recall"]))

        print("\n")

    for lab in labs_to_consider:
        print(lab)
        for criterion in criteria_used[lab]:
            print("{} : {}".format(criterion, ",".join(str(value) for key, value in sorted(
                zs_original_rating_counts[lab][criterion].items()))))
            print("Zero shot : {:.2f} Precision : [{} {:.2f}] Recall : [{} {:.2f}] [TP : {} FP : {} TN : {} FN : {}]".format(zs_criterion_wise_accuracies[lab][criterion], "(" + ",".join(str(round(value["precision"], 2)) for key, value in sorted(zs_rating_wise_metrics[lab][criterion].items())) + ")", zs_criterion_wise_metrics[lab][criterion]["precision"], "(" + ",".join(str(round(value["recall"], 2)) for key, value in sorted(zs_rating_wise_metrics[lab]
                  [criterion].items())) + ")", zs_criterion_wise_metrics[lab][criterion]["recall"], "(" + ",".join(str(value) for key, value in sorted(zs_tp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zs_fp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zs_tn[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zs_fn[lab][criterion].items())) + ")"))
            print("Few shot  : {:.2f} Precision [{} {:.2f}] Recall : [{} {:.2f}] [TP : {} FP : {} TN : {} FN : {}]".format(fs_criterion_wise_accuracies[lab][criterion], "(" + ",".join(str(round(value["precision"], 2)) for key, value in sorted(fs_rating_wise_metrics[lab][criterion].items())) + ")", fs_criterion_wise_metrics[lab][criterion]["precision"], "(" + ",".join(str(round(value["recall"], 2)) for key, value in sorted(fs_rating_wise_metrics[lab]
                  [criterion].items())) + ")", fs_criterion_wise_metrics[lab][criterion]["recall"], "(" + ",".join(str(value) for key, value in sorted(fs_tp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(fs_fp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(fs_tn[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(fs_fn[lab][criterion].items())) + ")"))
            print("SFT       : {:.2f} Precision [{} {:.2f}] Recall : [{} {:.2f}] [TP : {} FP : {} TN : {} FN : {}]".format(sft_criterion_wise_accuracies[lab][criterion], "(" + ",".join(str(round(value["precision"], 2)) for key, value in sorted(sft_rating_wise_metrics[lab][criterion].items())) + ")", sft_criterion_wise_metrics[lab][criterion]["precision"], "(" + ",".join(str(round(value["recall"], 2)) for key, value in sorted(sft_rating_wise_metrics[lab]
                  [criterion].items())) + ")",  sft_criterion_wise_metrics[lab][criterion]["recall"], "(" + ",".join(str(value) for key, value in sorted(sft_tp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(sft_fp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(sft_tn[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(sft_fn[lab][criterion].items())) + ")"))
            print("DPO       : {:.2f} Precision [{} {:.2f}] Recall : [{} {:.2f}] [TP : {} FP : {} TN : {} FN : {}]".format(dpo_criterion_wise_accuracies[lab][criterion], "(" + ",".join(str(round(value["precision"], 2)) for key, value in sorted(dpo_rating_wise_metrics[lab][criterion].items())) + ")",  dpo_criterion_wise_metrics[lab][criterion]["precision"], "(" + ",".join(str(round(value["recall"], 2)) for key, value in sorted(dpo_rating_wise_metrics[lab]
                  [criterion].items())) + ")", dpo_criterion_wise_metrics[lab][criterion]["recall"], "(" + ",".join(str(value) for key, value in sorted(dpo_tp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(dpo_fp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(dpo_tn[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(dpo_fn[lab][criterion].items())) + ")"))
            print("0R        : {:.2f} Precision [{} {:.2f}] Recall : [{} {:.2f}] [TP : {} FP : {} TN : {} FN : {}]".format(zr_criterion_wise_accuracies[lab][criterion], "(" + ",".join(str(round(value["precision"], 2)) for key, value in sorted(zr_rating_wise_metrics[lab][criterion].items())) + ")", zr_criterion_wise_metrics[lab][criterion]["precision"], "(" + ",".join(str(round(value["recall"], 2)) for key, value in sorted(zr_rating_wise_metrics[lab]
                  [criterion].items())) + ")", zr_criterion_wise_metrics[lab][criterion]["recall"], "(" + ",".join(str(value) for key, value in sorted(zr_tp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zr_fp[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zr_tn[lab][criterion].items())) + ")", "(" + ",".join(str(value) for key, value in sorted(zr_fn[lab][criterion].items())) + ")"))

            print("\n")

    if (save_path):
        accuracy_metrics = {"Zero Shot": zs_criterion_wise_accuracies, "Few Shot": fs_criterion_wise_accuracies,
                            "SFT": sft_criterion_wise_accuracies, "DPO": dpo_criterion_wise_accuracies, "0R": zr_criterion_wise_accuracies}
        precision_recall_metrics = {"Zero Shot": zs_criterion_wise_metrics, "Few Shot": fs_criterion_wise_metrics,
                                    "SFT": sft_criterion_wise_metrics, "DPO": dpo_criterion_wise_metrics, "0R": zr_criterion_wise_metrics}
        plot_criterion_wise_histograms(
            accuracy_metrics, save_path=save_path, metric_name="Accuracy")
        plot_criterion_wise_histograms(
            precision_recall_metrics, save_path=save_path, metric_name="Precision", metric_key="precision")
        plot_criterion_wise_histograms(
            precision_recall_metrics, save_path=save_path, metric_name="Recall", metric_key="recall")

    if (save_comparison_path):
        sorted_labs = sorted(zs_criterion_wise_accuracies.keys())

        zs_accuracies = [zs_lab_wise_macro_accuracies[lab]
                         for lab in sorted_labs]
        fs_accuracies = [fs_lab_wise_macro_accuracies[lab]
                         for lab in sorted_labs]
        sft_accuracies = [sft_lab_wise_macro_accuracies[lab]
                          for lab in sorted_labs]
        dpo_accuracies = [dpo_lab_wise_macro_accuracies[lab]
                          for lab in sorted_labs]

        zs_precision = [zs_lab_wise_metrics[lab]["precision"]
                        for lab in sorted_labs]
        fs_precision = [fs_lab_wise_metrics[lab]["precision"]
                        for lab in sorted_labs]
        sft_precision = [sft_lab_wise_metrics[lab]["precision"]
                         for lab in sorted_labs]
        dpo_precision = [dpo_lab_wise_metrics[lab]["precision"]
                         for lab in sorted_labs]

        zs_recall = [zs_lab_wise_metrics[lab]["recall"] for lab in sorted_labs]
        fs_recall = [fs_lab_wise_metrics[lab]["recall"] for lab in sorted_labs]
        sft_recall = [sft_lab_wise_metrics[lab]["recall"]
                      for lab in sorted_labs]
        dpo_recall = [dpo_lab_wise_metrics[lab]["recall"]
                      for lab in sorted_labs]

        bar_width = 0.1
        x = np.arange(len(sorted_labs))

        # plt.figure(figsize=(8,6))

        plt.bar(x - 3 * bar_width, zs_accuracies,
                width=2*bar_width, label='Zero-shot')
        plt.bar(x - bar_width, fs_accuracies,
                width=2*bar_width, label='Few-shot')
        plt.bar(x + bar_width, sft_accuracies, width=2*bar_width, label='SFT')
        plt.bar(x + 3 * bar_width, dpo_accuracies,
                width=2*bar_width, label='DPO')

        plt.xlabel('Lab number')
        plt.ylabel("Accuracy")
        plt.title(f'Comparison of accuracies of all approaches for all labs')
        plt.ylim(0, 1)
        plt.xticks(x, 1 + np.arange(len(sorted_labs)))
        plt.xticks(fontsize=8)
        plt.legend()

        # Save the plot
        plt.savefig(f"new_plots/accuracy/all_approaches_all_labs.png")
        plt.clf()

        plt.bar(x - 3 * bar_width, zs_precision,
                width=2*bar_width, label='Zero-shot')
        plt.bar(x - bar_width, fs_precision,
                width=2*bar_width, label='Few-shot')
        plt.bar(x + bar_width, sft_precision, width=2*bar_width, label='SFT')
        plt.bar(x + 3 * bar_width, dpo_precision,
                width=2*bar_width, label='DPO')

        plt.xlabel('Lab number')
        plt.ylabel("Precision")
        plt.title(f'Comparison of precision for all approaches for all labs')
        plt.ylim(0, 1)
        plt.xticks(x, 1 + np.arange(len(sorted_labs)))
        plt.xticks(fontsize=8)
        plt.legend()

        # Save the plot
        plt.savefig(f"new_plots/precision/all_approaches_all_labs.png")
        plt.clf()

        plt.bar(x - 3 * bar_width, zs_recall,
                width=2 * bar_width, label='Zero-shot')
        plt.bar(x - bar_width, fs_recall, width=2 *
                bar_width, label='Few-shot')
        plt.bar(x + bar_width, sft_recall, width=2 * bar_width, label='SFT')
        plt.bar(x + 3 * bar_width, dpo_recall,
                width=2 * bar_width, label='DPO')

        plt.xlabel('Lab number')
        plt.ylabel("Recall")
        plt.title(f'Comparison of recall for all approaches for all labs')
        plt.ylim(0, 1)
        plt.xticks(x, 1 + np.arange(len(sorted_labs)))
        plt.xticks(fontsize=8)
        plt.legend()

        # Save the plot
        plt.savefig(f"new_plots/recall/all_approaches_all_labs.png")
        plt.clf()


def plot_criterion_wise_histograms(metrics, save_path, metric_name="Accuracy", metric_key=None):
    metric_values = {}
    for method_name, method_metrics in metrics.items():
        metric_values[method_name] = []
        for lab_name in method_metrics.keys():
            if (metric_key):
                for criterion_name in method_metrics[lab_name]:
                    metric_values[method_name].append(
                        method_metrics[lab_name][criterion_name][metric_key])
            else:
                for criterion_name, metric_value in method_metrics[lab_name].items():
                    metric_values[method_name].append(metric_value)
        plt.hist(metric_values[method_name], bins=np.arange(
            11)*0.1, color="grey", edgecolor="black")
        plt.title(f"Histogram showing {metric_name} of various criteria")
        plt.xlabel(method_name + " " + metric_name)
        plt.ylabel("Number of criteria")
        plt.ylim(0, 16)
        img_save_path = os.path.join(save_path, metric_name)
        plt.savefig(os.path.join(img_save_path, method_name+".png"))
        plt.clf()


if __name__ == "__main__":
    # path = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/zero_shot_results_sigcse'
    # inspect_results(path)

    # zs_folder_name = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/zero_shot_results_sigcse/CP_00102_loop_cs101f22_LE01_A_Q2/Criterion 1.json"
    # fs_folder_name = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/few_shot_results_verbose"
    # sft_folder_name = None
    # dpo_folder_name = None

    # eval_labs_set_1 = ["CP_00116_sort_cs101f22_LE03_E_Q1", "CP_00214_var_cs101s23_lq01_d_q2", "CP_00110_loop_cs101f22_LE01_C_Q2", "CP_00112_fun_cs101f22_LE02_C_Q1", "CP_00315_var_cs101a23_lq01_c_q5"]

    # eval_labs_set_2 = ["CP_00104_loop_cs101f22_LE02_B_Q1", "CP_00111_array_cs101f22_LE02_B_Q2", "CP_00117_mat_cs101f22_LE03_B_Q2", "CP_00206_var_cs101s23_lq01_b_q2", "CP_00211_condition_cs101s23_LQ01_C_Q3"]

    # eval_labs_set_3 = ["CP_00101_loop_cs101f22_LE01_A_Q1", "CP_00106_loop_cs101f22_LE2_A_Q2", "CP_00107_sort_cs101f22_LE03_B_Q1","CP_00115_sort_cs101f22_LE03_C_Q1", "CP_00208_var_LQ01_B_Q4"]
 # //

    # compare_zs_fs(zs_folder_name, fs_folder_name, "comparison_results_new/zs_fs_both_simple_test_5_5.json")
    # analyse_zs_fs_results("comparison_results_new/zs_fs_both_simple_test_5_5.json", plot_name="zs_fs_both_simple", save_plot=0)

    # compare_all(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, "comparison_results_new/zs_simple_fs_simple_lora_simple_dpo_simple_model_1.json")

    # compare_all(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name)

    # analyse_all_results("comparison_results_new/zs_simple_fs_simple_lora_simple_dpo_simple_model_1.json", uncommon_criteria=1, eval_labs=eval_labs_set_1, use_eval=1, save_plot=0, plot_name="zs_fs_both_simple_sft_dpo_both_json_seen_common")

    # inspect_results(dpo_folder_name)

    # analyse_results(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, uncommon_criteria=1, eval_labs=eval_labs_set_1, use_eval=1, save_path="Histograms/set_1/unseen_uncommon")

    # analyse_results(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, uncommon_criteria=1, eval_labs=eval_labs_set_3, use_eval=1)

    # analyse_results(zs_folder_name, fs_folder_name, sft_folder_name, dpo_folder_name, common_criteria=1, eval_labs=eval_labs_set_1, zr_only_train=1)

    # //
    # analyse_all_results(zs_folder_name)

    # All new grading mela 2 codess
    labs = ['cs101a23_lq01_d_q3', 'cs101s23_lq01_c_q4', 'cs101a23_lq01_a_q2', 'cs101a23_lq01_d_q4', 'CP_00102_loop_cs101f22_LE01_A_Q2', 'cs101a23_lq01_a_q5', 'cs101a23_lq01_a_q3',
            'cs101a23_lq01_b_q4', 'cs101f22_lab05-compulsory2-series', 'CP_00105_sort_cs101f22_LE2_E_Q2', 'CP_00113_sort_cs101f22_LE03_A_Q1', 'CP_00108_loop_cs101f22_LE01_B_Q2']
    for lab in labs:
        path = '/raid/ganesh/nagakalyani/nagakalyani/autograding/SIGCSE/dataset/' + \
            lab+'/rubric_ratings.csv'
        grades_o = extract_all_original_grades(path)
        zs_path = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/zero_shot_results/' + lab+'/'
        fs_path = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/SIGCSE_Code/few_shot_results_verbose/' + lab+'/'
        rubric_path = '/raid/ganesh/nagakalyani/nagakalyani/autograding/SIGCSE/dataset/'+lab
        criteria = list(get_rubrics(rubric_path).keys())
        outputs = []
        for c in criteria:
            grades_llm = extract_llm_ratings(zs_path, c)
            # print(grades_llm)
            final = calculate_metrics(grades_o[c], grades_llm)
            out = final['accuracy']
            outputs.append(f'{c}:{out}')
        # print(final['accuracy'])
        print(lab, outputs)
