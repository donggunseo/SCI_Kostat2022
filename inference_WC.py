from dataset import prepare_WC_inference
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, AutoConfig, Trainer, AutoTokenizer
import os
import gc
gc.enable()
import numpy as np
import torch
import pandas as pd
from utils import seed_everything
import torch.nn as nn
from model import CustomModel


def inference(model_checkpoint):
    for dir in model_checkpoint:
        if os.path.isdir(dir)==False:
            print('no directory')
            return
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint[0])
    tokenized_test_dataset, class_df = prepare_WC_inference(tokenizer)
    print('length of test_dataset : ',len(tokenized_test_dataset))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold = len(model_checkpoint)
    training_args = TrainingArguments(per_device_eval_batch_size=128, output_dir = '../inference')
    all_predictions=0
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
    for fold in range(kfold):
        model_path = model_checkpoint[fold]
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        # model = CustomModel.from_pretrained(model_path, config=config)
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
    submission.to_csv(f'../submission/submission_WC.csv', index=False)
    # submission['logits'] = all_predictions.tolist()
    # submission.to_csv(f'../submission/submission_WC_forensemble.csv', index=False)

if __name__ == "__main__":
    seed_everything(42)
    model_checkpoint = [f'../best_model/roberta_large_WC_fold{fold}' for fold in range(5)]
    inference(model_checkpoint)