from dataset import prepare_BC
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoConfig, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import concatenate_datasets
import wandb
import os
from utils import seed_everything
import argparse
from sklearn.metrics import accuracy_score

def train(neg = 3, kfold=5):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_dataset_list, tokenizer = prepare_BC(neg = 3, kfold=kfold)
    for fold in range(kfold):
        valid_dataset = kfold_tokenized_dataset_list[fold]
        train_dataset = concatenate_datasets([kfold_tokenized_dataset_list[i] for i in range(kfold) if i!=fold])
        config.num_labels = 2
        config = AutoConfig.from_pretrained('klue/roberta-large')
        model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=config)
        training_args = TrainingArguments(
            output_dir= f'../output/roberta_large_neg{neg}_fold{fold}',
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            per_device_train_batch_size = 32,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 1,
            learning_rate = 1e-5,
            weight_decay = 0.1,
            num_train_epochs = 4,
            warmup_ratio = 0.06,
            logging_strategy = 'steps',
            logging_dir = f'../log/roberta_large_neg{neg}_fold{fold}',
            logging_steps = 100,
            save_total_limit = 1,
            seed = 42,
            dataloader_num_workers = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'accuracy',
            group_by_length =True,
            report_to = 'wandb',
        )
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            return {'eval_accuracy' : acc*100}
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
        trainer=Trainer(
            model,
            training_args,
            train_dataset = train_dataset,
            eval_dataset = valid_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        run = wandb.init(project='kostat', entity='donggunseo', name=f'roberta_large_neg{neg}_fold{fold}')
        trainer.train()
        run.finish()
        trainer.save_model(f'../best_model/roberta_large_neg{neg}_fold{fold}')
        trainer.save_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg', type=int, default=3, help='decide the number of negative samples')
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    train(args.neg, args.kfold)
