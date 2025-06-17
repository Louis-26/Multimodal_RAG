#!/bin/bash
#$ -q gpu.q@@a100
#$ -l gpu=1
#$ -l h_rt=72:00:00
#$ -N classify_mvent2
#$ -o /home/hltcoe/amartin/SCALE/metadata-classification/logs
#$ -e /home/hltcoe/amartin/SCALE/metadata-classification/logs


eval "$(conda shell.bash hook)"

conda activate metadata-roberta_1

cd /home/hltcoe/amartin/SCALE/metadata-classification

# sh classify_metadata.sh

# sh classify_metadata_farsi.sh

python eval_roberta.py \
    --data /exp/amartin/data_for_metadata/hf_data_dict_unlabeled_info_mvent2.json \
    --save_dir /exp/amartin/data_for_metadata/output_preds_full_scrape/mvent2