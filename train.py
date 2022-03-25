from dataset import prepare
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoConfig, DataCollatorWithPadding
from datasets import concatenate_datasets
import wandb
import os
import numpy as np
from utils import seed_everything, post_processing
from trainer import CustomTrainer
import argparse
from sklearn.metrics import accuracy_score, f1_score

def train(neg=1, kfold = 5):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_file_list = [f'../neg{neg}_csv_{kfold}fold/train_neg{neg}_fold{i}.csv' for i in range(kfold)]
    valid_file_list = [f'../validation_csv_{kfold}fold/validation_fold{i}.csv' for i in range(kfold)]
    kfold_tokenized_train_dataset, kfold_tokenized_valid_dataset, tokenizer, N_LABELS = prepare(
        neg = neg, 
        k = kfold, 
        train_file_list = train_file_list, 
        valid_file_list = valid_file_list
        )
    for fold in range(kfold):
        valid_datasets = kfold_tokenized_valid_dataset[fold]
        valid_gt = list(valid_datasets['class_num'])
        valid_gt = [valid_gt[i] for i in range(0, len(valid_gt), neg+1)]
        train_datasets = concatenate_datasets([kfold_tokenized_train_dataset[i].flatten_indices() for i in range(kfold) if i!=fold])
        config = AutoConfig.from_pretrained('klue/roberta-large')
        config.num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config = config)
        training_args = TrainingArguments(
            output_dir = f'../output/roberta_large_neg{neg}_fold{fold}',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size = 32,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 1,
            learning_rate = 1e-5,
            weight_decay = 0.01,
            max_grad_norm = 10,
            num_train_epochs = 5,
            warmup_ratio = 0.1,
            logging_strategy = 'steps',
            logging_dir = '../log/roberta_large_neg{neg}_fold{fold}',
            logging_steps = 50,
            save_strategy = 'epoch',
            save_total_limit = 1,
            seed = 42,
            dataloader_num_workers = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'accuracy',
            group_by_length = True,
            report_to = 'wandb',
        )
        def compute_metrics(eval_gt, eval_preds):
            assert len(eval_gt)==len(eval_preds), 'mismatch between gt and preds'
            accuracy = accuracy_score(y_true = eval_gt, y_pred = eval_preds)
            f1 = f1_score(y_true = eval_gt, y_pred = eval_preds, average = 'macro')
            return {'eval_accuracy' : accuracy, 'f1' : f1}
        data_collator = DataCollatorWithPadding(tokenizer)
        trainer=CustomTrainer(
            model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=valid_datasets,
            eval_gt=valid_gt,
            data_collator=data_collator,
            tokenizer=tokenizer,
            post_process_function = post_processing,
            compute_metrics=compute_metrics
        )
        run = wandb.init(project='kostat', entity='donggunseo', name='roberta_large_neg{neg}_fold{fold}')
        trainer.train()
        run.finish()
        trainer.save_model('../best_model/roberta_large_neg{neg}_fold{fold}')
        trainer.save_state('../log/roberta_large_neg{neg}_fold{fold}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg', type=int, default=1, help='decide the number of negative samples')
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    train(args.neg, args.kfold)
