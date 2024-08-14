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
        print(rows)
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
                        print(i, idx, rows[i][idx])
                        reasonings[criterion][student_id] = rows[i][idx]
                        if rows[i][idx] == 'No Comments':
                            reasonings[criterion][student_id] = rows[3][idx]
                        # reasonings[criterion][student_id] = criterion_rating_titles[criterion][idx - start_idx]

        return reasonings


def extract_original_criterion_reasoning(grades_file_path, criterion_name):
    grades_file_path = os.path.join(grades_file_path, 'rubric_ratings.csv')
    all_grades = extract_all_original_grades(grades_file_path)

    return all_grades[criterion_name]


if __name__ == "__main__":
    lab_path = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/datasetreasoning27/CP_00102_loop_cs101f22_LE01_A_Q2"
    lab_grades_path = os.path.join(lab_path, "rubric_ratings.csv")
    original_grades = extract_all_original_grades(lab_grades_path)
    all_criteria = get_rubrics(lab_path)

    # Get the original TA grades for that lab
    original_reasonings = extract_all_original_reasonings(lab_grades_path)
    print(all_criteria)
    print(12313123)
    print(original_reasonings)
    print(12313123)
    # Repeat for all criteria
    for criterion in all_criteria.keys():
        # Description for that particular criterion
        criterion_desc = all_criteria[criterion]["description"]
        # Rating descriptions for that particular criterion
        options = all_criteria[criterion]["ratings"]
        print(criterion)
        # Get the Grades And Rubrics related to specific criterion
        criterion_original_grades = original_grades[criterion]
        criterion_original_reasonings = original_reasonings[criterion]
        print(criterion_original_reasonings)
        print()

    original_grade = 'A'
    options['A'] = 'This is poor comments'
    original_reasoning = 'Comments are poor'
    chosen_response = '''{"answer" : ''' + \
        f'''"{original_grade}. {options[original_grade]} , "reasoning" : {original_reasoning}"''' + '''} </s>'''
    print(chosen_response)
