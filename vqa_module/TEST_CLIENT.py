import os
import requests
import json
from datetime import datetime
from tqdm import tqdm
from __init__ import IMG_PATH

url = "http://192.168.100.141:2503/process"
dir_path = IMG_PATH
all_image_paths = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))][:500]

if not os.path.exists('../outputs'):
    os.makedirs('../outputs')
output_folder = f"../outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_folder)

params = {
    'batch_size': 100,
    'question_weights': '0.1 0.2 0.1 0.2 0.1 0.1 0.2',
    'threshold': 0.8
}
data = {
    'questions': [
        'Using yes or no, are there people in this image?',
        'Using yes or no, is this image in a studio, with a plain color background?',
        'Using yes or no, is this image an illustration?',
        'Using yes or no, are there people of races other than Asian and Caucasian in this image?',
        'Using yes or no, are there anyone above the age of 50 in this image?',
        'Using yes or no, are there both men and women in the image?',
        'Using yes or no, does the image exude a stressful atmosphere?',
    ],
    'expected_answers': ['yes', 'no', 'no', 'no', 'no', 'yes', 'yes'],
}

# ---------- Save params and data ---------- #
with open(f"{output_folder}/params.json", "w") as f:
    json.dump(params, f)
with open(f"{output_folder}/data.json", "w") as f:
    json.dump(data, f)

# ---------- Send request in batches and save responses ---------- #
for i in tqdm(range(0, len(all_image_paths), 100)):
    start_idx = i
    end_idx = i + 100 if len(all_image_paths) - start_idx >= 100 else len(all_image_paths)
    image_paths = all_image_paths[start_idx:end_idx]

    files = [("images", (os.path.basename(image_path), open(image_path, "rb"), "images")) for image_path in image_paths]

    response = requests.post(url, files=files, data=data, params=params)
    with open('f"{output_folder}/response.json', 'a') as json_file:
        json.dump(response.json(), json_file)
        json_file.write('\n')
