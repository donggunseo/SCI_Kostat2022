## 일반 Sequence Classification training을 수행하는 코드

from dataset import prepare_WC
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoConfig, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import concatenate_datasets
import wandb
import os
from utils import seed_everything
import argparse
from sklearn.metrics import accuracy_score, f1_score

def train(kfold=5):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_dataset_list, tokenizer = prepare_WC(kfold=kfold)
    # 반복문을 돌면서 지정된 폴드를 검증 데이터셋으로, 나머지 폴드들을 훈련 데이터셋으로 사용
    for fold in range(kfold):
        valid_dataset = kfold_tokenized_dataset_list[fold]
        train_dataset = concatenate_datasets([kfold_tokenized_dataset_list[i] for i in range(kfold) if i!=fold])
        # 훈련에 사용하는 config, model은 모두 Huggingface library에서 불러와 사용
        config = AutoConfig.from_pretrained('klue/roberta-large')
        config.num_labels = 232
        model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=config)
        # 훈련과정에서 customize하는 argument들
        training_args = TrainingArguments(
            output_dir= f'../output/roberta_large_WC_fold{fold}',
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
        ## 검증 후 metric을 확인하기 위한 함수
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            return {'eval_accuracy' : acc*100, 'eval_f1' : f1 * 100}
        ## tokenize과정에서 padding을 하지 않았기 때문에 batch마다 dynamic padding하기 위한 클래스
        data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
        ## Huggingface에서 제공하는 모델 훈련 및 검증이 쉽게 가능한 Trainer클래스를 활용해 학습, 검증, 로그 저장, 모델 저장 등의 과정을 수행함
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
        ## 학습과정과 여러 지표들을 그래프로 확인하기 위해 wandb API를 활용
        run = wandb.init(project='kostat', entity='donggunseo', name=f'roberta_large_WC_fold{fold}')
        trainer.train()
        run.finish()
        ## 각 에폭의 model checkpoint 중 가장 성능이 우수한 모델을 해당 폴드에 대한 대표모델로 따로 저장
        trainer.save_model(f'../best_model/roberta_large_WC_fold{fold}')
        ## 학습 로그를 저장하는 코드
        trainer.save_state()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=5, help='decide the number of fold for stratify kfold')
    args = parser.parse_args()
    seed_everything(42)
    train(args.kfold)