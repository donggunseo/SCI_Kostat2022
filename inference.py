from dataset import prepare_WC_inference
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, AutoConfig, Trainer, AutoTokenizer
import os
import gc
gc.enable()
import numpy as np
import torch
import pandas as pd
from utils import seed_everything
from model import CustomModel
from tqdm import tqdm


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
        print("model load from : ", model_path)
        config = AutoConfig.from_pretrained(model_path)
        if fold==0 or fold==1:
            print(f"best model for fold{fold} is Original Sequence classification model")
            model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        else:
            print(f"best model for fold{fold} is multi-dropout Sequence classification model")
            model = CustomModel.from_pretrained(model_path, config=config)
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
        print(f"Aggregate fold{fold} model logit")
        predictions = predictions.astype(np.float32)
        predictions = predictions/kfold
        all_predictions+=predictions
        torch.cuda.empty_cache()
        gc.collect()
    preds = all_predictions.argmax(-1)
    print("shape of label prediction after argmax",preds.shape)
    submission = pd.read_csv('../input/답안 작성용 파일.csv', encoding='CP949')
    first = []
    second = []
    third = []
    for pred in tqdm(preds, desc = "add prediction to Dataframe"):
        a = list(class_df[class_df['class_num']==pred]['1st'])[0]
        b = list(class_df[class_df['class_num']==pred]['2nd'])[0]
        c = list(class_df[class_df['class_num']==pred]['3rd'])[0]
        first.append(a)
        second.append(b)
        third.append(c)
    submission['digit_1'] = first
    submission['digit_2'] = second
    submission['digit_3'] = third
    print("save result as csv")
    os.makedirs('../submission', exist_ok=True)
    submission.to_csv(f'../submission/submission.csv', index=False)

if __name__ == "__main__":
    seed_everything(42)
    ## Fold별 결과를 보고 best choice만 골라서 path에 담기
    model_checkpoint = [f'../best_model/roberta_large_WC_fold{fold}' for fold in range(0,2)]
    model_checkpoint1 = [f'../best_model/roberta_large_WC_MD_fold{fold}' for fold in range(2,5)]
    model_checkpoint.extend(model_checkpoint1)
    inference(model_checkpoint)