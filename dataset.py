from preprocess import combine
from create_kfold import create_kfold
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import seed_everything
import os

def generate_random_neg_index(neg, total_len, row_class_num):
    r = [i for i in range(total_len) if i!=row_class_num]
    num_label = random.sample(r, neg)
    num_label.append(row_class_num)
    num_label.sort()
    return num_label

def prepare_negsample_df(query_df=None, class_df=None, neg=1, kfold=5, file_list = None):
    fold_df_list=[]
    if file_list!=None and len([f for f in file_list if os.path.isfile(f)])==len(file_list): 
        for file in tqdm(file_list, desc='reading csv from disk'):
            df = pd.read_csv(file)
            assert len(df)==((1000000/kfold) * (neg+1)), 'load wrong csv, put correct csv file'
            fold_df_list.append(df)
        return fold_df_list
    else:
        total_len = len(class_df)
        dir_name = f'../neg{neg}_csv_{kfold}fold'
        os.makedirs(dir_name, exist_ok=True)
        for i in range(kfold):
            df = query_df[query_df['kfold']==i].reset_index(drop=True)
            id_list = []
            text_list = []
            label_list = []
            kfold_list = []
            class_text_list = []
            class_num_list = []
            for j in tqdm(range(len(df))):
                row = df.iloc[j]
                row_class_num = row['class_num']
                num_label = generate_random_neg_index(neg, total_len, row_class_num)
                id_list.extend([row['AI_id'] for _ in range(len(num_label))])
                text_list.extend([row['query_text'] for _ in range(len(num_label))])
                kfold_list.extend([row['kfold'] for _ in range(len(num_label))])
                class_num_list.extend([row_class_num for _ in range(len(num_label))])
                for k in num_label:
                    class_text_list.append(list(class_df[class_df['class_num']==k]['class_text'])[0])
                    if k == row_class_num:
                        label_list.append(1)
                    else:
                        label_list.append(0)
            new_df = pd.DataFrame({'AI_id' : id_list, 'query_text' : text_list, 'class_num' : class_num_list, 'kfold' : kfold_list, 'class_text' : class_text_list, 'label' : label_list})
            new_df.to_csv(f'{dir_name}/train_neg{neg}_fold{i}.csv')
            fold_df_list.append(new_df)
        return fold_df_list

def prepare(neg=1, k=5, train_file_list = None, valid_file_list = None, hf_train_dataset_list = None, hf_valid_dataset_list = None):
    kfold_tokenized_train_dataset = []
    kfold_tokenized_valid_dataset = []
    if hf_train_dataset_list!=None and hf_valid_dataset_list!=None and len([f for f in hf_train_dataset_list if os.path.isdir(f)])==len(hf_train_dataset_list) and len([f for f in hf_valid_dataset_list if os.path.isdir(f)])==len(hf_valid_dataset_list):
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
        for fold in range(k):
            tokenized_train_dataset = load_from_disk(hf_train_dataset_list[fold])
            tokenized_valid_dataset = load_from_disk(hf_valid_dataset_list[fold])
            kfold_tokenized_train_dataset.append(tokenized_train_dataset)
            kfold_tokenized_valid_dataset.append(tokenized_valid_dataset)
        return kfold_tokenized_train_dataset, kfold_tokenized_valid_dataset, tokenizer
    query_df, class_df = combine()
    query_df = create_kfold(query_df, k=k)
    train_fold_df_list = prepare_negsample_df(query_df, class_df, neg = neg, kfold = k, file_list = train_file_list)
    valid_fold_df_list = prepare_negsample_df(query_df, class_df, neg = 9, kfold = k, file_list = valid_file_list)
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    def train_mapping(examples):
        encoding = tokenizer(
            examples['query_text'],
            examples['class_text'],
            truncation=True,
            padding=False,
            max_length=512,
            return_token_type_ids=False
        )
        encoding['labels'] = examples['label']
        return encoding
    def valid_mapping(examples):
        encoding = tokenizer(
            examples['query_text'],
            examples['class_text'],
            truncation=True,
            padding=False,
            max_length=512,
            return_token_type_ids=False
        )
        encoding['class_num'] = examples['class_num']
        return encoding
    kfold_tokenized_train_dataset = []
    kfold_tokenized_valid_dataset = []
    for fold in range(k):
        train_datasets = Dataset.from_pandas(train_fold_df_list[fold])
        valid_datasets = Dataset.from_pandas(valid_fold_df_list[fold])
        tokenized_train_dataset = train_datasets.map(train_mapping, batched=True, batch_size=10000, remove_columns=train_datasets.column_names)
        tokenized_valid_dataset = valid_datasets.map(valid_mapping, batched=True, batch_size=10000, remove_columns=valid_datasets.column_names)
        tokenized_train_dataset.flatten_indices()
        tokenized_valid_dataset.flatten_indices()
        tokenized_train_dataset.save_to_disk(f'../train_dataset/train_dataset_neg{neg}_fold{k}')
        tokenized_valid_dataset.save_to_disk(f'../valid_dataset/valid_dataset_neg9_fold{k}')
        kfold_tokenized_train_dataset.append(tokenized_train_dataset)
        kfold_tokenized_valid_dataset.append(tokenized_valid_dataset)
    return kfold_tokenized_train_dataset, kfold_tokenized_valid_dataset, tokenizer

def prepare_inference(test_file= None, model_checkpoint = None):
    query_df, class_df = combine(type='test')
    if test_file!=None and os.path.isfile(test_file): 
        df = pd.read_csv(test_file)
        assert len(df)==100000*232, 'load wrong csv, put correct csv file'
    else:
        dir_name = f'../test_csv'
        os.makedirs(dir_name, exist_ok=True)
        total_len = len(class_df)
        id_list = []
        query_text_list = []
        class_text_list = []
        class_num_list = []
        for i in tqdm(range(len(query_df))):
            row = query_df.iloc[i]
            id_list.extend([row['AI_id'] for _ in range(total_len)])
            query_text_list.extend([row['query_text'] for _ in range(total_len)])
            class_text_list.extend(class_df['class_text'])
            class_num_list.extend(class_df['class_num'])
        df = pd.DataFrame({'AI_id' : id_list, 'query_text' : query_text_list, 'class_num' : class_num_list, 'class_text' : class_text_list})
        df.to_csv(f'{dir_name}/test.csv')
    test_datasets = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def valid_mapping(examples):
        encoding = tokenizer(
            examples['query_text'],
            examples['class_text'],
            truncation=True,
            padding=False,
            max_length=512,
            return_token_type_ids=False
        )
        encoding['class_num'] = examples['class_num']
        return encoding
    tokenized_test_datasets = test_datasets.map(valid_mapping, batched=True, batch_size=10000, remove_columns=test_datasets.column_names)
    return tokenized_test_datasets, class_df, tokenizer, total_len




