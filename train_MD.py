## 직접 설계한 Multi-dropout model로 Sequence Classification train을 수행하는 코드
## train_WC.py와 model부분을 제외하고는 대부분 같음

from dataset import prepare_WC
from transformers import TrainingArguments, AutoConfig, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import concatenate_datasets
import wandb
import os
from utils import seed_everything
import argparse
from sklearn.metrics import accuracy_score, f1_score
from model import CustomModel

def train(kfold=5):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_dataset_list, tokenizer = prepare_WC(kfold=kfold)
    for fold in range(kfold):
        valid_dataset = kfold_tokenized_dataset_list[fold]
        train_dataset = concatenate_datasets([kfold_tokenized_dataset_list[i] for i in range(kfold) if i!=fold])
        config = AutoConfig.from_pretrained('klue/roberta-large')
        config.num_labels = 232
        model = CustomModel.from_pretrained('klue/roberta-large', config=config)
        training_args = TrainingArguments(
            output_dir= f'../output/roberta_large_WC_MD_fold{fold}',
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            per_device_train_batch_size = 128,
            per_device_eval_batch_size = 128,
            gradient_accumulation_steps = 1,
            learning_rate = 5e-5,
            weight_decay = 0.1,
            num_train_epochs = 4,
            warmup_ratio = 0.1,
            logging_strategy = 'steps',
            logging_steps = 50,
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
            f1 = f1_score(labels, preds, average='macro')
            return {'eval_accuracy' : acc*100, 'eval_f1' : f1 * 100}
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
        run = wandb.init(project='kostat', entity='donggunseo', name=f'roberta_large_WC_MD_fold{fold}')
        trainer.train()
        run.finish()
        trainer.save_model(f'../best_model/roberta_large_WC_MD_fold{fold}')
        trainer.save_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    train(args.kfold)