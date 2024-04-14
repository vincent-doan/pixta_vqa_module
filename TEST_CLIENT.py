import os
import argparse
import requests
import json
from datetime import datetime
from tqdm import tqdm

from metrics import calculate_metrics

def main():

    # COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Input host')
    parser.add_argument('--port', help='Input port', default=2503)
    parser.add_argument('--total_images', help='Input total images', default=-1, type=int)
    parser.add_argument('--batch_size', help='Input batch size', default=100, type=int)
    parser.add_argument('--query_details', help='Input path to query details', default='./query_details_req1.json')
    parser.add_argument('--true_labels', help='Input path to true labels', default='./labels/labels_for_req1.json')
    parser.add_argument('--image_folder', help='Input path to image folder', default='./final_data_10k')
    args = parser.parse_args()

    # LOAD IMAGES
    url = f"http://{args.host}:{args.port}/process"
    dir_path = args.image_folder
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
    with open(args.query_details, 'r') as f:
        query_details = json.load(f)
    params = query_details['params']
    data = query_details['data']

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
    
    # SAVE METRICS
    accepted_image_ids = [image["image_id"].split('.')[0] for image in accepted_images]
    accuracy, precision, recall, f1, true_positive_image_ids = calculate_metrics(accepted_image_ids, args.true_labels, total=args.total_images)

    # SAVE OVERALL STATISTICS
    with open(f"{output_folder}/stats_overall.json", "w") as f:
        json.dump({
            'process_time_taken': round(process_time_taken, 2),
            'total_time_taken': round(total_time_taken, 2),
            'accepted_images': accepted_images,
            'accepted_image_ids': sorted(accepted_image_ids),
            'true_positive_image_ids': sorted(true_positive_image_ids),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }, f, indent=4)

if __name__ == '__main__':
    main()
