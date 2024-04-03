import os
import requests
from . import IMG_PATH

url = "http://192.168.100.141:2503/process"
dir_path = IMG_PATH
image_paths = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))]
files = [("images", (os.path.basename(image_path), open(image_path, "rb"), "images")) for image_path in image_paths]

params = {
    'batch_size': 100,
    'question_weights': '0.1 0.2 0.1 0.2 0.1 0.1 0.2',
    'threshold': 0.8
}

data = {
    'questions': ['Using yes or no, are there people in this image?',
                  'Using yes or no, is this image in a studio, with a plain color background?',
                  'Using yes or no, is this image an illustration?',
                  'Using yes or no, are there people of races other than Asian and Caucasian in this image?',
                  'Using yes or no, are there anyone above the age of 50 in this image?',
                  'Using yes or no, are there both men and women in the image?',
                  'Using yes or no, does the image exude a stressful atmosphere?',
    ],
    'expected_answers': ['yes', 'no', 'no', 'no', 'no', 'yes', 'yes'],
}

response = requests.post(url, files=files, data=data, params=params)
print(response.json())