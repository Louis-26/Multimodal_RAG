import os 
import json 
import argparse 

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Format metadata')
    parser.add_argument(
        '--metadata',
        type=str,
        default='/exp/amartin/scale24/metadata-classification/scale24_json',
        help='Path to metadata directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/exp/amartin/data_for_metadata',
        help='Path to output directory'
    )
    args = parser.parse_args()
    return args



def format_metadata_string(metadata_json):
    metadata_string = ''

    language = metadata_json['lang']
    caption = metadata_json['caption']
    category = metadata_json['category']
    yt_meta_dict = metadata_json['yt_meta_dict']['info']

    title = yt_meta_dict['title']
    resolution = yt_meta_dict['resolution']
    fps = yt_meta_dict['fps']
    tags = yt_meta_dict['tags']
    was_live = yt_meta_dict['was_live']
    filesize_approx = yt_meta_dict['filesize_approx']

    
    metadata_string = f"Language: {language} </s> Caption: {caption} </s> Category: {category} </s> Title: {title} </s> Resolution: {resolution} </s> FPS: {fps} </s> Tags: {tags} </s> Was Live: {was_live} </s> Filesize Approx: {filesize_approx} </s>"
    # print(metadata_string)
    return metadata_string


def main():
    args = parse_args()

    data = {}
    # for file in os.listdir(args.metadata):
    for file in tqdm(os.listdir(args.metadata)):
        if file.endswith('.json'):
            with open(os.path.join(args.metadata, file), 'r') as f:
                metadata = json.load(f)
                id = file.split('.json')[0]
                metadata_string = format_metadata_string(metadata)
                data[id] = {
                    'metadata_string': metadata_string,
                    'video_type': 'Unknown',
                }

    with open(os.path.join(args.output_dir, 'unlabeled_info_mvent2.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()