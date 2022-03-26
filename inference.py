from dataset import prepare_inference
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoConfig, DataCollatorWithPadding, Trainer
import os
import gc
gc.enable()
import numpy as np
import torch
import pandas as pd
from utils import seed_everything
import argparse


def inference(model_checkpoint):
    for direc in model_checkpoint:
        if os.path.isdir(direc)==False:
            print('no directory')
            return 
    test_file = '../test_csv/test.csv'
    tokenized_test_datasets, class_df, tokenizer, N_LABELS = prepare_inference(test_file, model_checkpoint[0])
    kfold = len(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    training_args = TrainingArguments(per_device_eval_batch_size=32, output_dir = '../inference')
    all_predictions = 0
    for fold in kfold:
        model_path =model_checkpoint[fold]
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config = config)
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = None,
            eval_dataset=tokenized_test_datasets,
            tokenizer = tokenizer,
            data_collator = data_collator,
        )
        predictions, _, _ = trainer.predict(test_dataset = tokenized_test_datasets)
        print("shape of prediction", predictions.shape)
        predictions = predictions.astype(np.float32)
        predictions = predictions/5
        all_predictions+=predictions
        torch.cuda.empty_cache()
        gc.collect()
    all_predictions = np.array(all_predictions)
    softmax = torch.nn.Softmax(dim=-1)
    all_predictions = torch.tensor(all_predictions)
    pred_score = softmax(all_predictions)
    pred_score = pred_score.view(-1, 232, 2)
    pred_score = pred_score.numpy()
    pred_score = pred_score[:,:,1]
    preds = np.argmax(pred_score, axis=-1)
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
    submission.to_csv(f'../submission/submission_neg1.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg', type=int, default=1, help='decide the number of negative samples')
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    model_checkpoint = [f'../best_model/roberta_large_neg{args.neg}_fold{i}' for i in range(args.kfold)]
    inference(model_checkpoint)


