from dataset2 import prepare
from transformers import AutoModelForMultipleChoice, TrainingArguments, AutoConfig, Trainer
from datasets import concatenate_datasets
import wandb
import os
import numpy as np
from utils import seed_everything
import argparse
from sklearn.metrics import accuracy_score, f1_score
from model import DataCollatorForMultipleChoice

def train(choice=10, kfold=5):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_dataset_list, tokenizer = prepare(choice=choice, kfold=kfold)
    for fold in range(kfold):
        valid_dataset = kfold_tokenized_dataset_list[fold]
        train_dataset = concatenate_datasets([kfold_tokenized_dataset_list[i] for i in range(kfold) if i!=fold])
        config = AutoConfig.from_pretrained('klue/roberta-large')
        model = AutoModelForMultipleChoice.from_pretrained('klue/roberta-large', config=config)
        training_args = TrainingArguments(
            output_dir= f'../output/roberta_large_choice{choice}_fold{fold}',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size = 3,
            per_device_eval_batch_size = 3,
            gradient_accumulation_steps = 1,
            learning_rate = 1e-5,
            weight_decay = 0.01,
            num_train_epochs = 5,
            warmup_ratio = 0.06,
            logging_strategy = 'steps',
            logging_dir = f'../log/roberta_large_choice{choice}_fold{fold}',
            logging_steps = 50,
            save_strategy = 'epoch',
            save_total_limit = 1,
            seed = 42,
            dataloader_num_workers = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'accuracy',
            report_to = 'wandb',
        )
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            return {'eval_accuracy' : acc, 'eval_f1':f1}
        data_collator = DataCollatorForMultipleChoice(tokenizer)
        trainer=Trainer(
            model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics
        )
        run = wandb.init(project='kostat', entity='donggunseo', name=f'roberta_large_choice{choice}_fold{fold}')
        trainer.train()
        run.finish()
        trainer.save_model(f'../best_model/roberta_large_choice{choice}_fold{fold}')
        trainer.save_state(f'../training_state/roberta_large_choice{choice}_fold{fold}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=int, default=10, help='decide the number of choice')
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    train(args.choice, args.kfold)
