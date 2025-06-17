import os 
import json 
import argparse

import jsonlines
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Format metadata')
    parser.add_argument(
        '--metadata_json_dir',
        type=str,
        default='/exp/scale24/features/metadata/internvid_to_classify',
        help='Path to metadata directory'
    )
    parser.add_argument(
        '--test_judgements',
        type=str,
        default='/exp/rkriz/code/SCALE2024/infrastructure/queries/test_mapped_judgments_pt2_updated_evalai.jsonl',
        help='Path to judgements file'
    )
    # parser.add_argument(
    #     '--train_judgements',
    #     type=str,
    #     default='/exp/rkriz/code/SCALE2024/infrastructure/queries/train_mapped_judgments_pt2_updated_evalai.jsonl',
    #     help='Path to judgements file'
    # )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/exp/scale24/features/metadata/internvid_to_classify_hf_format',
        help='Path to output directory'
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    judgement_dict = {}
    for line in jsonlines.open(args.test_judgements):
        orig_id = line['original_doc_id']
        judgement = line['video_type']
        if 'raw' in judgement.lower():
            judgement = 'Raw'
        else:
            judgement = 'Professional'
        judgement_dict[orig_id] = judgement

    
    for filename in os.listdir(args.metadata_json_dir):
        hf_data_dict = {
            "ids": [],
            "metadata": [],
            "video_type": []
        }
        if filename.endswith('.json'):
            metadata_json = json.load(open(os.path.join(args.metadata_json_dir, filename)))
            for metadata_dict in metadata_json:
                full_name = list(metadata_dict.keys())[0]
                last_part = full_name.split('/')[-1]
                name = last_part.split('.')[0]
                metadata_string = metadata_dict[full_name]["metadata_string"]
                hf_data_dict["ids"].append(name)
                hf_data_dict["metadata"].append(metadata_string)
                hf_data_dict["video_type"].append("Unknown")
            with open(os.path.join(args.output_dir, f'{filename}'), 'w') as f:
                json.dump(hf_data_dict, f)


                
    # with open(os.path.join(args.output_dir, 'test_metadata.json'), 'w') as f:
    #     json.dump(hf_data_dict, f)
               
                



if __name__ == '__main__':
    main()

