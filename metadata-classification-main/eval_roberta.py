import os 
import json
import argparse



import torch
from transformers import (
    AutoTokenizer, 
    XLMRobertaForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from functools import partial
import evaluate
import torch.nn as nn
import numpy as np

f1 = evaluate.load('f1')

def preprocess(examples: LazyBatch, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True)


def gen(data):
    for i, (id, metadata, video_type) in enumerate(zip(data['ids'], data['metadata'], data['video_type'])):
        
        if video_type == 'Professional':
            label = 0
        elif video_type == 'Raw':
            label = 1
        else:
            label = 1
        yield {
            'id': id,
            'text': metadata,
            'label': label
        }

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    # print(predictions, labels)
    # predictions = np.argmax
    # predictions = predictions[:, 0]
    # print(predictions, labels)
    # print(AUC.compute(predictions=predictions, references=labels))
    return f1.compute(predictions=predictions, references=labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='/exp/amartin/data_for_metadata/hf_data_dict_unlabeled_info_all_correctfor.json',
        help='Path to metadata file'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/home/hltcoe/amartin/SCALE/metadata-classification/ckpts/best_model',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/exp/scale24/features/metadata/internvid_classified',
        help='Path to metadata file'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    seed = 42
    with open(args.data, 'r') as f:
        data = json.load(f)

    id2label = {
        0: 'Professional',
        1: 'Raw'
    }
    label2id = {
        'Professional': 0,
        'Raw': 1
    }

    model = XLMRobertaForSequenceClassification.from_pretrained(
        args.ckpt, num_labels=2, id2label=id2label, label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')


    data_dataset = Dataset.from_generator(partial(gen, data))
    tokenized_data = data_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True) # load_from_cache_file=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_batch_size = 128
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # set best model metric to be f1
        metric_for_best_model='f1',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    os.makedirs(args.save_dir, exist_ok=True)


    # save predictions to file
    predictions = trainer.predict(tokenized_data)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    print(predictions)
    predictions = {} # doc_id -> label 
    # print(tokenized_data)
    print(pred_labels)
    # breakpoint()
    for i, instance in enumerate(tokenized_data):
        doc_id = instance['id']
        label = pred_labels[i]
        if label == 0:
            label = 'Professional'
        else:
            label = 'Raw'
        predictions[doc_id] = label

    # with open(os.path.join(args.save_dir, 'unlabeled_mvent_predictions.json'), 'w') as f:
    #     json.dump(predictions, f)
        
    new_file_name = args.data.split('/')[-1].split('.json')[0] + '_predictions.json'
    print(f"Saving predictions to {new_file_name}")
    with open(os.path.join(args.save_dir, new_file_name), 'w') as f:
        json.dump(predictions, f)
    

    
if __name__ == '__main__':
    main()

