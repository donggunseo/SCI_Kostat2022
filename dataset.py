from preprocess import combine
from create_kfold import create_kfold
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import seed_everything
import os
from itertools import chain

def prepare_WC(kfold=5):
    seed_everything(42)
    kfold_dataset_list = [f'../dataset/WC_fold{i}' for i in range(kfold)]
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    kfold_tokenized_dataset_list = []
    if len([f for f in kfold_dataset_list if os.path.isdir(f)])==len(kfold_dataset_list):
        print('train dataset file already exist! \n load dataset')
        for fold in range(kfold):
            tokenized_dataset = load_from_disk(kfold_dataset_list[fold])
            kfold_tokenized_dataset_list.append(tokenized_dataset)
        return kfold_tokenized_dataset_list, tokenizer
    print('Generating train dataset from beginning')
    query_df, _ = combine('train')
    query_df = create_kfold(df = query_df, kfold=kfold)
    for fold in range(kfold):
        df = query_df[query_df['kfold']==fold].reset_index(drop=True)
        query_text_list = list(df['query_text'])
        print(f'tokenize for fold{fold}')
        encoding = tokenizer(
            query_text_list,
            truncation=True,
            padding=False,
            max_length=310,
            return_token_type_ids=False,
        )
        encoding['labels'] = list(df['class_num'])
        assert len(encoding['input_ids']) == len(encoding['labels']), 'x and label length mismatch'
        dataset = Dataset.from_dict(encoding)
        dataset.flatten_indices()
        print(f'save train dataset to {kfold_dataset_list[fold]}')
        dataset.save_to_disk(kfold_dataset_list[fold])
        kfold_tokenized_dataset_list.append(dataset)
    return kfold_tokenized_dataset_list, tokenizer

        
def prepare_WC_inference(tokenizer):
    test_dataset_path = f'../dataset/test_dataset_WC'
    query_df, class_df = combine('test')
    if os.path.isdir(test_dataset_path):
        print('test dataset file already exist! \n load dataset')
        test_tokenized_dataset = load_from_disk(test_dataset_path)
        return test_tokenized_dataset, class_df
    print('Generating test dataset from beginning')
    data_length = len(query_df)
    query_text_list = list(query_df['query_text'])
    print(f'tokenize')
    encoding = tokenizer(
        query_text_list,
        truncation=True,
        padding=False,
        max_length=310,
        return_token_type_ids=False,
    )
    assert len(encoding['input_ids'])==data_length, 'reshape size mismatch problem'
    test_tokenized_dataset = Dataset.from_dict(encoding)
    test_tokenized_dataset.flatten_indices()
    print(f'save test dataset to {test_dataset_path}')
    test_tokenized_dataset.save_to_disk(test_dataset_path)
    return test_tokenized_dataset, class_df


        


