import os 
import json 
import argparse
from tqdm import tqdm

# /home/hltcoe/amartin/SCALE/metadata-classification/data/test_info.json

json_data = json.load(open('/exp/amartin/data_for_metadata/unlabeled_info_mvent2.json', 'r'))

new_dict = {
    "ids": [],
    "metadata": [],
    "video_type": []
}


for key, value in tqdm(json_data.items()):
    new_dict["ids"].append(key)
    new_dict["metadata"].append(value['metadata_string'])
    new_dict["video_type"].append(value['video_type'])


with open('/exp/amartin/data_for_metadata/hf_data_dict_unlabeled_info_mvent2.json', 'w') as f:
    json.dump(new_dict, f)


