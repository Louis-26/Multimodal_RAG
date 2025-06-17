#!/bin/bash
#$ -l h_rt=72:00:00
#$ -N format_data_dict
#$ -o /home/hltcoe/amartin/SCALE/metadata-classification/logs/
#$ -e /home/hltcoe/amartin/SCALE/metadata-classification/logs/


eval "$(conda shell.bash hook)"

conda activate metadata-roberta_1

cd /home/hltcoe/amartin/SCALE/metadata-classification

python scripts/format_unlabeled_data.py --metadata /exp/amartin/scale24/info/internvid/all_jsons


python scripts/format_as_data_dict.py