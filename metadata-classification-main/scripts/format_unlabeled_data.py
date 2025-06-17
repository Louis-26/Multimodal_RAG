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

    language = ''
    if 'language' in metadata_json:
        language = metadata_json['language']
    else:
        print("No language key in metadata_json")
    
    caption = ''
    if 'description' in metadata_json:
        caption = metadata_json['description']
    else:
        print("No description key in metadata_json")

    category = ''
    if 'categories' in metadata_json:
        category = metadata_json['categories']
    else:
        print("No categories key in metadata_json")

    title = ''
    if 'title' in metadata_json:
        title = metadata_json['title']
    else:
        print("No title key in metadata_json")
        
    resolution = ''
    if 'resolution' in metadata_json:
        resolution = metadata_json['resolution']
    else:
        print("No resolution key in metadata_json")
    
    fps = ''
    if 'fps' in metadata_json:
        fps = metadata_json['fps']
    else:
        print("No fps key in metadata_json")

    tags = ''
    if 'tags' in metadata_json:
        tags = metadata_json['tags']
    else:
        print("No tags key in metadata_json")

    was_live = ''
    if 'was_live' in metadata_json:
        was_live = metadata_json['was_live']
    else:
        print("No was_live key in metadata_json")

    filesize_approx = ''
    if 'filesize_approx' in metadata_json:
        filesize_approx = metadata_json['filesize_approx']
    else:
        print("No filesize_approx key in metadata_json")

    
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

    with open(os.path.join(args.output_dir, 'unlabeled_info_all_correct.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()