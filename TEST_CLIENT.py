import os
import argparse
import requests
import json
from datetime import datetime
from tqdm import tqdm

def main():

    # COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Input host')
    parser.add_argument('--port', help='Input port', default=2503)
    parser.add_argument('--total_images', help='Input total images', default=-1, type=int)
    parser.add_argument('--batch_size', help='Input batch size', default=100, type=int)
    parser.add_argument('--question_weights', help='Input question weights', default=None)
    parser.add_argument('--threshold', help='Input threshold', default=None, type=float)
    args = parser.parse_args()

    # LOAD IMAGES
    url = f"http://{args.host}:{args.port}/process"
    dir_path = './imgs'
    if args.total_images == -1:
        all_image_paths = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))]
    else:
        all_image_paths = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))][:args.total_images]

    # CREATE OUTPUT FOLDER
    os.makedirs('./outputs', exist_ok=True)
    output_folder = f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_folder)
    os.makedirs(f"{output_folder}/responses")

    # DEFINE PARAMS AND DATA
    params = {
        'question_weights': args.question_weights,
        'threshold': args.threshold
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

    with open(f"{output_folder}/params.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(f"{output_folder}/data.json", "w") as f:
        json.dump(data, f, indent=4)

    # SEND REQUESTS IN BATCHES
    process_time_taken = 0
    total_time_taken = 0
    accepted_images = []
    for i in tqdm(range(0, len(all_image_paths), args.batch_size)):
        start_idx = i
        end_idx = i + args.batch_size if len(all_image_paths) - start_idx >= args.batch_size else len(all_image_paths)
        image_paths = all_image_paths[start_idx:end_idx]
        
        files = [("images", (os.path.basename(image_path), open(image_path, "rb"), "images")) for image_path in image_paths]
        response = requests.post(url, files=files, data=data, params=params)
        
        # Overall statistics
        process_time_taken += response.json()['time_taken']
        total_time_taken += response.elapsed.total_seconds()
        accepted_images.extend(response.json()['accepted_images'])
        
        # Save each response
        with open(f"{output_folder}/responses/response_{i // args.batch_size}.json", 'a') as json_file:
            json.dump(response.json(), json_file, indent=4)
            json_file.write('\n')

    # SAVE OVERALL STATISTICS
    with open(f"{output_folder}/stats_overall.json", "w") as f:
        json.dump({
            'process_time_taken': round(process_time_taken, 2),
            'total_time_taken': round(total_time_taken, 2),
            'accepted_images': accepted_images
        }, f, indent=4)

if __name__ == '__main__':
    main()
