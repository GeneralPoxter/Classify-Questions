"""
Developer: Jason Liu
Commnets: Based on setup code from Atith Gandhi
"""

import json
import os
import random
import requests

COLLEGE_LIM = 5000
HS_LIM = 5000
SPLIT = 8000

# Define labels
college = ["College", "easy_college", "regular_college", "hard_college"]
hs = [
    "HS",
    "hard_high_school",
    "easy_high_school",
    "regular_high_school",
    "national_high_school",
]

# Download qanta dataset
dirname = os.path.dirname(__file__)
if not os.path.isfile(os.path.join(dirname, "data/qanta.json")):
    print("Downloading data into data/qanta.json...")
    res = requests.get(
        "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/qanta.train.2018.04.18.json"
    )
    with open("data/qanta.json", "w") as qanta:
        qanta.write(res.text)

# Load and randomize order of questions in the dataset
f = open("data/qanta.json")
data = json.load(f)
random.shuffle(data["questions"])

labeled_data = []
hs_count = 0
college_count = 0

# Label questions as either 'college' or 'hs'
for question in data["questions"]:
    if question["difficulty"] in college and college_count < COLLEGE_LIM:
        labeled_data.append({"text": question["text"], "difficulty": "college"})
        college_count += 1
    elif question["difficulty"] in hs and hs_count < HS_LIM:
        labeled_data.append({"text": question["text"], "difficulty": "hs"})
        hs_count += 1

# Split labeled data into training and testing datasets
print("Splitting labeled data into data/qanta_train.json and data/qanta_test.json...")
with open("data/qanta_train.json", "w") as outfile:
    json.dump({"questions": labeled_data[:SPLIT]}, outfile)
with open("data/qanta_test.json", "w") as outfile:
    json.dump({"questions": labeled_data[SPLIT:]}, outfile)
