from preprocess import combine
from create_kfold import create_kfold
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import seed_everything
import os
from itertools import chain

def generate_random_index(choice, gt):
    r = [i for i in range(232) if i!=gt]
    num_label = random.sample(r, choice-1)
    num_label.append(gt) ## gt는 가장 마지막에 들어있으므로 모든 Instance의 label = choice-1(9)!!
    return num_label

def prepare_MC(choice = 10, kfold=5):
    seed_everything(42)
    kfold_dataset_list = [f'../dataset/{choice}choice_fold{i}' for i in range(kfold)]
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    kfold_tokenized_dataset_list = []
    if len([f for f in kfold_dataset_list if os.path.isdir(f)])==len(kfold_dataset_list):
        print('train dataset file already exist! \n load dataset')
        for fold in range(kfold):
            tokenized_dataset = load_from_disk(kfold_dataset_list[fold])
            kfold_tokenized_dataset_list.append(tokenized_dataset)
        return kfold_tokenized_dataset_list, tokenizer
    print('Generating train dataset from beginning')
    query_df, class_df = combine('train')
    query_df = create_kfold(df = query_df, kfold=kfold)
    class_text_list = list(class_df['class_text'])
    for fold in range(kfold):
        df = query_df[query_df['kfold']==fold].reset_index(drop=True)
        data_length = len(df)
        query_text_list = [query for query in df['query_text'] for _ in range(0,choice)]
        choice_class_list = []
        for class_num in tqdm(df['class_num'], desc=f'generating random choice fold{fold}'):
            random_choice = generate_random_index(choice=choice, gt = class_num)
            random_choice_text = [class_text_list[c] for c in random_choice]
            choice_class_list.extend(random_choice_text)
        assert len(query_text_list)==len(choice_class_list), 'query list and class list length mismatch'
        print(f'tokenize for fold{fold}')
        encoding = tokenizer(
            query_text_list,
            choice_class_list,
            truncation=True,
            padding=False,
            max_length=310,
            return_token_type_ids=False,
        )
        key_list= list(encoding.keys())
        encoding = {key:[encoding[key][i:i+choice] for i in range(0,data_length*choice,choice)] for key in key_list}
        assert len(encoding['input_ids'])==data_length, 'reshape size mismatch problem'
        encoding['labels'] = [choice-1 for _ in range(data_length)]
        dataset = Dataset.from_dict(encoding)
        dataset.flatten_indices()
        print(f'save train dataset to {kfold_dataset_list[fold]}')
        dataset.save_to_disk(kfold_dataset_list[fold])
        kfold_tokenized_dataset_list.append(dataset)
    return kfold_tokenized_dataset_list, tokenizer

def prepare_MC_inference(tokenizer):
    test_dataset_path = f'../dataset/test_dataset_MC'
    query_df, class_df = combine('test')
    if os.path.isdir(test_dataset_path):
        print('test dataset file already exist! \n load dataset')
        test_tokenized_dataset = load_from_disk(test_dataset_path)
        return test_tokenized_dataset, class_df
    print('Generating test dataset from beginning')
    data_length = len(query_df)
    class_text_list = list(class_df['class_text']) * data_length
    query_text_list = [query for query in query_df['query_text'] for _ in range(0,232)]
    assert len(query_text_list)==len(class_text_list), 'query list and class list length mismatch'
    print(f'tokenize')
    encoding = tokenizer(
        query_text_list,
        class_text_list,
        truncation=True,
        padding=False,
        max_length=310,
        return_token_type_ids=False,
    )
    key_list= list(encoding.keys())
    encoding = {key:[encoding[key][i:i+232] for i in range(0,data_length*232,232)] for key in key_list}
    assert len(encoding['input_ids'])==data_length, 'reshape size mismatch problem'
    test_tokenized_dataset = Dataset.from_dict(encoding)
    test_tokenized_dataset.flatten_indices()
    print(f'save test dataset to {test_dataset_path}')
    test_tokenized_dataset.save_to_disk(test_dataset_path)
    return test_tokenized_dataset, class_df

def prepare_BC(neg = 3, kfold=5):
    seed_everything(42)
    kfold_dataset_list = [f'../dataset/BC_neg{neg}_fold{i}' for i in range(kfold)]
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    kfold_tokenized_dataset_list = []
    if len([f for f in kfold_dataset_list if os.path.isdir(f)])==len(kfold_dataset_list):
        print('train dataset file already exist! \n load dataset')
        for fold in range(kfold):
            tokenized_dataset = load_from_disk(kfold_dataset_list[fold])
            kfold_tokenized_dataset_list.append(tokenized_dataset)
        return kfold_tokenized_dataset_list, tokenizer
    print('Generating train dataset from beginning')
    query_df, class_df = combine('train')
    query_df = create_kfold(df = query_df, kfold=kfold)
    class_text_list = list(class_df['class_text'])
    for fold in range(kfold):
        df = query_df[query_df['kfold']==fold].reset_index(drop=True)
        data_length = len(df)
        query_text_list = [query for query in df['query_text'] for _ in range(0,neg+1)]
        choice_class_list = []
        for class_num in tqdm(df['class_num'], desc=f'generating random choice fold{fold}'):
            random_choice = generate_random_index(choice=neg+1, gt = class_num)
            random_choice_text = [class_text_list[c] for c in random_choice]
            choice_class_list.extend(random_choice_text)
        assert len(query_text_list)==len(choice_class_list), 'query list and class list length mismatch'
        print(f'tokenize for fold{fold}')
        encoding = tokenizer(
            query_text_list,
            choice_class_list,
            truncation=True,
            padding=False,
            max_length=310,
            return_token_type_ids=False,
        )
        encoding['labels'] = list(chain(*[[0]*neg+[1] for _ in range(data_length)]))
        assert len(encoding['input_ids'])==data_length*(neg+1), 'input_ids size mismatch problem'
        assert len(encoding['labels'])==data_length*(neg+1), 'labels size mismatch problem'
        dataset = Dataset.from_dict(encoding)
        dataset.flatten_indices()
        print(f'save train dataset to {kfold_dataset_list[fold]}')
        dataset.save_to_disk(kfold_dataset_list[fold])
        kfold_tokenized_dataset_list.append(dataset)
    return kfold_tokenized_dataset_list, tokenizer

def prepare_BC_inference(tokenizer):
    test_dataset_path = f'../dataset/test_dataset_BC'
    query_df, class_df = combine('test')
    if os.path.isdir(test_dataset_path):
        print('test dataset file already exist! \n load dataset')
        test_tokenized_dataset = load_from_disk(test_dataset_path)
        return test_tokenized_dataset, class_df
    print('Generating test dataset from beginning')
    data_length = len(query_df)
    class_text_list = list(class_df['class_text']) * data_length
    query_text_list = [query for query in query_df['query_text'] for _ in range(0,232)]
    assert len(query_text_list)==len(class_text_list), 'query list and class list length mismatch'
    print(f'tokenize')
    encoding = tokenizer(
        query_text_list,
        class_text_list,
        truncation=True,
        padding=False,
        max_length=310,
        return_token_type_ids=False,
    )
    assert len(encoding['input_ids'])==data_length*232, 'reshape size mismatch problem'
    test_tokenized_dataset = Dataset.from_dict(encoding)
    test_tokenized_dataset.flatten_indices()
    print(f'save test dataset to {test_dataset_path}')
    test_tokenized_dataset.save_to_disk(test_dataset_path)
    return test_tokenized_dataset, class_df

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


        


