

# /exp/scale24/features/metadata/internvid_to_classify_hf_format
for i in /exp/scale24/features/metadata/internvid_to_classify_hf_format/*; do
    echo "Processing $i"
    # /home/hltcoe/amartin/SCALE/metadata-classification/eval_roberta.py
    python eval_roberta.py --data $i --ckpt /home/hltcoe/amartin/SCALE/metadata-classification/ckpts/iter2/current-best
done