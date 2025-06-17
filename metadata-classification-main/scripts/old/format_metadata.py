import os 
import json
import argparse


import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description='Format metadata')
    parser.add_argument(
        '--judgements',
        type=str,
        default='/exp/rkriz/code/SCALE2024/infrastructure/queries/test_mapped_judgments_pt2_updated_evalai.jsonl',
        help='Path to judgements file'
    )
    parser.add_argument(
        '--mapping_csv',
        type=str,
        default='/exp/amartin/scale24/metadata-classification/video_id_mapping_v2.csv',
        help='Path to mapping csv'
    )
    parser.add_argument(
        '--train_metadata',
        type=str,
        default='/exp/rkriz/data/scale2024/video/train_judged_metadata/',

    )
    parser.add_argument(
        '--test_metadata',
        type=str,
        default='/exp/rkriz/data/scale2024/video/test_judged_metadata/',
    )
    # parser.add_argument(
    #     '--metadata_dir',
    #     type=str,
    #     default='/exp/amartin/scale24/metadata-classification/scale24_json',
    #     help='Path to metadata directory'
    # )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/hltcoe/amartin/SCALE/metadata-classification/data',
        help='Path to output directory'
    )
    args = parser.parse_args()
    return args



def format_metadata_string(metadata_json):
    metadata_string = ''
    # "lang": "Unknown",
    # "caption": "the flags flying in front of a chinese embassy",
    # "category": "News & Politics",

    # metadata_string = f"Caption: {metadata_json['caption']} </s> Category: {metadata_json['category']} </s> Language: {metadata_json['lang']} </s>"
    print(metadata_json.keys())
    # dict_keys(['lang', 'caption', 'category', 'url', 'key', 'status', 'error_message', 'yt_meta_dict'])

    language = metadata_json['lang']
    caption = metadata_json['caption']
    category = metadata_json['category']
    yt_meta_dict = metadata_json['yt_meta_dict']['info']
    print(yt_meta_dict.keys())

    title = yt_meta_dict['title']
    resolution = yt_meta_dict['resolution']
    fps = yt_meta_dict['fps']
    tags = yt_meta_dict['tags']
    was_live = yt_meta_dict['was_live']
    filesize_approx = yt_meta_dict['filesize_approx']
    
    metadata_string = f"Language: {language} </s> Caption: {caption} </s> Category: {category} </s> Title: {title} </s> Resolution: {resolution} </s> FPS: {fps} </s> Tags: {tags} </s> Was Live: {was_live} </s> Filesize Approx: {filesize_approx} </s>"

    return metadata_string

def main():
    args = parse_args()

    judgements = {}
    num_judgements = 0
    with open(args.judgements, 'r') as f:
        for line in f:
            data = json.loads(line)
            judgements[data['original_doc_id']] = data
            num_judgements += 1

    mappings = {}
    source_mappings = {}
    with open(args.mapping_csv, 'r') as f:
        for line in f:
            # scale_id,orig_id,source
            scale_id, orig_id, source = line.strip().split(',')
            mappings[orig_id] = scale_id
            source_mappings[orig_id] = source

    data = {}
    type_counts = {}

    for id, judgement in judgements.items():
        # check if id in train or test 
        if os.path.exists(os.path.join(args.train_metadata, id + '.json')):
            metadata_file = os.path.join(args.train_metadata, id + '.json')
        elif os.path.exists(os.path.join(args.test_metadata, id + '.json')):
            metadata_file = os.path.join(args.test_metadata, id + '.json')
        else:
            print(f"metadata file not found for {id}")
            print(f"source: {source_mappings[id]}")
            print(f"Has csv: {os.path.exists(os.path.join(args.test_metadata, id + '.csv'))}")
            continue

        metadata_json = json.load(open(metadata_file, 'r'))

        metadata_string = format_metadata_string(metadata_json)
        # print(metadata_string)
        vid_type = ''
        if judgement['video_type'] == 'Professional' or judgement['video_type'] == 'Edited':
            vid_type = 'Professional'
        elif judgement['video_type'] == 'Diet Raw' or judgement['video_type'] == 'Raw':
            vid_type = 'Raw'

        data[id] = {
            'video_type': vid_type,
            'metadata_string': metadata_string,
            'judgement': judgement,
            'metadata': metadata_json,
        }
        if judgement['video_type'] not in type_counts:
            type_counts[judgement['video_type']] = 0
        type_counts[judgement['video_type']] += 1

    with open(os.path.join(args.output_dir, 'test_info.json'), 'w') as f:
        json.dump(data, f)

    print(len(data))
    print(type_counts)
    print(num_judgements)

if __name__ == '__main__':
    main()

