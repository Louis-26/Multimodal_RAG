import json
import argparse


# /home/hltcoe/amartin/SCALE/metadata-classification/data/iter2/test_metadata.json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='/home/hltcoe/amartin/SCALE/metadata-classification/data/iter2/test_metadata.json',
        help='Path to metadata file'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.data, 'r') as f:
        data = json.load(f)
    # count the number of each class in the video_type field
    video_type_counts = {}
    for video_type in data['video_type']:
        if video_type in video_type_counts:
            video_type_counts[video_type] += 1
        else:
            video_type_counts[video_type] = 1

    print(video_type_counts)

    class_0_weight = (video_type_counts['Raw'] + video_type_counts['Professional']) / (video_type_counts['Professional'] * 2)
    class_1_weight = (video_type_counts['Raw'] + video_type_counts['Professional']) / (video_type_counts['Raw'] * 2)

    print(f'class 0 weight: {class_0_weight}')
    print(f'class 1 weight: {class_1_weight}')

if __name__ == '__main__':
    main()
