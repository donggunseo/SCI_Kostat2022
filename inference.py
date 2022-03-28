from dataset import prepare_inference
from transformers import AutoModelForMultipleChoice, TrainingArguments, AutoConfig, Trainer, AutoTokenizer
import os
import gc
gc.enable()
import numpy as np
import torch
import pandas as pd
from utils import seed_everything
import argparse
from model import DataCollatorForMultipleChoice
from itertools import chain
from model import Model

def inference(model_checkpoint):
    for dir in model_checkpoint:
        if os.path.isdir(dir)==False:
            print('no directory')
            return
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint[0])
    tokenized_test_dataset, class_df = prepare_inference(tokenizer)
    # def flatten(examples):
    #     encoding = {k:list(chain(*v)) for k,v in examples.items()}
    #     return encoding
    # tokenized_test_dataset = tokenized_test_dataset.map(flatten, batched = True, batch_size=10000, remove_columns=tokenized_test_dataset.column_names)
    print('length of test_dataset : ',len(tokenized_test_dataset))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold = len(model_checkpoint)
    training_args = TrainingArguments(per_device_eval_batch_size=1, output_dir = '../inference')
    all_predictions=0
    data_collator = DataCollatorForMultipleChoice(tokenizer)
    for fold in range(kfold):
        model_path = model_checkpoint[fold]
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForMultipleChoice.from_pretrained(model_path, config=config)
        # model = Model.from_pretrained(model_path, config = config)
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = None,
            eval_dataset=tokenized_test_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator
        )
        predictions,_,_ = trainer.predict(test_dataset = tokenized_test_dataset)
        print("shape of prediction", predictions.shape)
        predictions = predictions.astype(np.float32)
        predictions = predictions/kfold
        all_predictions+=predictions
        torch.cuda.empty_cache()
        gc.collect()
    print(all_predictions.shape)
    preds = all_predictions.argmax(-1)
    # preds = all_predictions.reshape(len(tokenized_test_dataset), 232)
    # preds = preds.argmax(-1)
    print(preds.shape)
    submission = pd.read_csv('../input/답안 작성용 파일.csv', encoding='CP949')
    first = []
    second = []
    third = []
    for pred in preds:
        a = list(class_df[class_df['class_num']==pred]['1st'])[0]
        b = list(class_df[class_df['class_num']==pred]['2nd'])[0]
        c = list(class_df[class_df['class_num']==pred]['3rd'])[0]
        first.append(a)
        second.append(b)
        third.append(c)
    submission['digit_1'] = first
    submission['digit_2'] = second
    submission['digit_3'] = third
    os.makedirs('../submission', exist_ok=True)
    submission.to_csv(f'../submission/submission_choice10.csv')

if __name__ == "__main__":
    seed_everything(42)
    # model_checkpoint = [f'../best_model/roberta_large_choice10_fold{fold}' for fold in range(5)]
    model_checkpoint=['../output/roberta_large_choice10_fold0/checkpoint-6666']
    inference(model_checkpoint)