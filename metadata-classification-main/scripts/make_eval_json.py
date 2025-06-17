import os 
import json 
import argparse
import tarfile

from tqdm import tqdm

# Cameron's Farsi data:
# /exp/scale24/data/yt-aparat/yt-aparat-metadata-cleaned

def parse_args():
    parser = argparse.ArgumentParser(description='Format metadata')
    parser.add_argument(
        '--metadata',
        type=str,
        default='/exp/scale24/data/internvid/info',
        help='Path to metadata directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/exp/scale24/features/metadata/internvid_to_classify',
        help='Path to output directory'
    )
    args = parser.parse_args()
    return args

def format_metadata_string(metadata_json):
     #    'title',
            #    'formats', 
            #    'description', 
            #    'duration', 
            #    'tags', 
            #    'duration_string',
            #    'was_live', 
            #    'format', 
            #    'format_id', 
            #    'filesize_approx',  
            #    'resolution', 
            #    'fps', 
            #    'aspect_ratio',
    title = ''
    if 'title' in metadata_json:
        title = metadata_json['title']
    description = ''
    if 'description' in metadata_json:
        description = metadata_json['description']
    duration = ''
    if 'duration' in metadata_json:
        duration = metadata_json['duration']
    tags = ''
    if 'tags' in metadata_json:
        tags = metadata_json['tags']
    duration_string = ''
    if 'duration_string' in metadata_json:
        duration_string = metadata_json['duration_string']
    was_live = ''
    if 'was_live' in metadata_json:
        was_live = metadata_json['was_live']
    filesize_approx = ''
    if 'filesize_approx' in metadata_json:
        filesize_approx = metadata_json['filesize_approx']
    resolution = ''
    if 'resolution' in metadata_json:
        resolution = metadata_json['resolution']
    fps = ''
    if 'fps' in metadata_json:
        fps = metadata_json['fps']
    aspect_ratio = ''
    if 'aspect_ratio' in metadata_json:
        aspect_ratio = metadata_json['aspect_ratio']

    metadata_string = f"Title: {title} </s> Description: {description} </s> Duration: {duration} </s> Tags: {tags} </s> Duration String: {duration_string} </s> Was Live: {was_live} </s> Filesize Approx: {filesize_approx} </s> Resolution: {resolution} </s> FPS: {fps} </s> Aspect Ratio: {aspect_ratio} </s>"

    # metadata_string = f"Title: {metadata_json['title']} </s> Description: {metadata_json['description']} </s> Duration: {metadata_json['duration']} </s> Tags: {metadata_json['tags']} </s> Duration String: {metadata_json['duration_string']} </s> Was Live: {metadata_json['was_live']} </s> Filesize Approx: {metadata_json['filesize_approx']} </s> Resolution: {metadata_json['resolution']} </s> FPS: {metadata_json['fps']} </s> Aspect Ratio: {metadata_json['aspect_ratio']} </s>"
    # print(metadata_string)
    return metadata_string

def main():
    args = parse_args()

    for file in tqdm(os.listdir(args.metadata)):
        if file.endswith('.tar.gz'):
            shard_metadata = []
            tar = tarfile.open(os.path.join(args.metadata, file), 'r:gz')
            for member in tar.getmembers():
                if member.name.endswith('.json'):
                    metadata_dict = {}
                    f = tar.extractfile(member)
                    metadata = json.load(f)
                    id = member.name.split('.json')[0]
                    metadata_string = format_metadata_string(metadata)
                    metadata_dict[id] = {
                        'metadata_string': metadata_string,
                    }
                    shard_metadata.append(metadata_dict)
            tar.close()
            with open(os.path.join(args.output_dir, file.split('.tar.gz')[0] + '.json'), 'w') as f:
                json.dump(shard_metadata, f, indent=4)



if __name__ == '__main__':
    main()

