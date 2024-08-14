import json
import os
import sys
import csv

dir = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27"

for lab in os.listdir(dir):
    # lab = "CP_00101_loop_cs101f22_LE01_A_Q1"
    labpath = os.path.join(dir, lab)
    rubric_path = os.path.join(labpath, 'rubrics.json')
    with open(rubric_path, 'r') as f:
        data = json.load(f)

    for criterion in data:
        for rating in criterion['Ratings']:
            if rating['title'] == 'A ':
                rating['title'] = 'A'
            rating['description'] = rating['description'].strip()
            rating['title'] = rating['title'].strip()
            criterion['description'] = criterion['description'].strip()
            criterion['title'] = criterion['title'].strip()

    with open(rubric_path, 'w') as f:
        json.dump(data, f, indent=4)

    csv_path = os.path.join(labpath, 'rubric_ratings.csv')
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # Clean spaces and tabs from the first row
    rows[0] = [cell.strip() for cell in rows[0]]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
