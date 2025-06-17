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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        type=str,
        default='/home/hltcoe/amartin/SCALE/metadata-classification/data/iter2/test_metadata.json',
        help='Path to metadata file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xlm-roberta-large',
        help='Path to model'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/hltcoe/amartin/SCALE/metadata-classification/ckpts/iter2',
    )
    args = parser.parse_args()
    return args

def preprocess(examples: LazyBatch, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True)


def gen(data):
    for i, (id, metadata, video_type) in enumerate(zip(data['ids'], data['metadata'], data['video_type'])):
        
        if video_type == 'Professional':
            label = 0
        else:
            label = 1
        yield {
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



class RobertaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # BCEWithLogitsLoss weight weight on the positive class
        # pos_weight = torch.tensor([100.0]).to('cuda')
        weights = torch.tensor([0.581, 3.58]).to('cuda')
        criterion = nn.CrossEntropyLoss(weight=weights)
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        # print(outputs.logits, labels)
        logits = outputs.logits.view(-1, 2)
        # print(logits, labels)
        # breakpoint()
        loss = criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
        args.model, num_labels=2, id2label=id2label, label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    data_dataset = Dataset.from_generator(partial(gen, data))
    tokenized_data = data_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, load_from_cache_file=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # set best model metric to be f1
        metric_for_best_model='f1',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed=seed,
    )

    trainer = RobertaTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    os.makedirs(args.output_dir, exist_ok=True)

    trainer.train()
    trainer.evaluate()

    # save best model 
    os.makedirs(os.path.join(args.output_dir, 'best_model'), exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, 'best_model'))



    
    




if __name__ == '__main__':
    main()