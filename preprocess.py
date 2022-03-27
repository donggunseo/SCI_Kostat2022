import pandas as pd
from tqdm import tqdm

def preprocess_query(type='train'):
    if type=='train':
        file_name = '../input/1. 실습용자료.txt'
    elif type=='test':
        file_name = '../input/2. 모델개발용자료.txt'
    with open(file_name, 'r', encoding='CP949') as f:
        data = f.read()
    data = data.split('\n')
    column_names = data[0].split('|')
    df_dict = [[] for _ in range(len(column_names))]
    for i in range(len(data)):
        if i==0:
            continue
        else:
            element = data[i].split('|')
            if len(column_names)!= len(element):
                continue
            else:
                for i in range(len(column_names)):
                    df_dict[i].append(element[i])
    df = pd.DataFrame({column_names[i] : df_dict[i] for i in range(len(column_names))})
    df = make_query_format(df)
    return df

def preprocess_class():
    class_df = pd.read_excel('../input/한국표준산업분류(10차)_국문.xlsx')
    new_header = class_df.iloc[0]
    class_df = class_df[1:]
    class_df.columns=new_header
    class_df.drop([1], axis=0, inplace=True)
    class_df.columns = ['1st', '1st_text', '2nd', '2nd_text', '3rd', '3rd_text', '4th', '4th_text', '5th', '5th_text']
    columns = list(class_df.columns)
    for column in columns:  
        class_df[column] = class_df[column].fillna(method = 'ffill')
    class_df = make_class_format(class_df)
    class_2nd_list = []
    for t in class_df['2nd']:
        if t[0]=='0':
            class_2nd_list.append(t[1:])
        else:
            class_2nd_list.append(t)
    class_3rd_list = []
    for t in class_df['3rd']:
        if t[0]=='0':
            class_3rd_list.append(t[1:])
        else:
            class_3rd_list.append(t)
    class_df['2nd']=class_2nd_list
    class_df['3rd']=class_3rd_list
    return class_df

def make_query_format(df):
    final_query = []
    for i, row in tqdm(df.iterrows(), desc = 'making query format'):
        query_text= []
        if row['text_obj']!='':
            query_text.append(row['text_obj'])
        if row['text_mthd']!='':
            query_text.append(row['text_mthd'])
        if row['text_deal']!='':
            query_text.append(row['text_deal'])
        query_text = ' '.join(query_text)
        final_query.append(query_text)
    df['query_text'] = final_query
    return df

def make_class_format(df):
    label_list = list(df['3rd_text'].unique())
    final_class= []
    for item in tqdm(label_list, desc='making class format'):
        df_temp = df[df['3rd_text']==item]
        detail_label= list(df_temp['5th_text'])
        if len(detail_label)==1:
            if detail_label[0]!=item:
                detail_label = detail_label[0] + ' 등을 포함하는 '+ item
            else:
                detail_label = detail_label[0]
        else:
            if item in detail_label:
                detail_label.remove(item)
                if len(detail_label)==1:
                    detail_label = detail_label[0] + ' 등을 포함하는 '+ item
                else:
                    detail_label = ', '.join(detail_label) + ' 등을 포함하는 '+ item
            else:
                detail_label = ', '.join(detail_label) + ' 등을 포함하는 '+ item
        final_class.append(detail_label)
    df.drop(['4th', '4th_text', '5th', '5th_text'], axis =1, inplace=True)
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    df['class_text'] = final_class
    df['class_num'] = [i for i in range(len(final_class))]
    return df

def combine(type='train'):
    query_df = preprocess_query(type)
    class_df = preprocess_class()
    if type=='test':
        query_df.drop(['digit_1', 'digit_2', 'digit_3', 'text_obj', 'text_mthd', 'text_deal'], axis=1, inplace=True)
        class_df.drop(['1st_text', '2nd_text', '3rd_text'], axis=1, inplace=True)
        return query_df, class_df
    elif type=='train':
        label_list=[]
        for i, row in tqdm(query_df.iterrows(), desc='adding class_num to query_df'):
            label = row['digit_3']
            label_num = int(class_df[class_df['3rd']==label]['class_num'])
            label_list.append(label_num)
        query_df['class_num']=label_list
        query_df.drop(['digit_1', 'digit_2', 'digit_3', 'text_obj', 'text_mthd', 'text_deal'], axis=1, inplace=True)
        class_df.drop(['1st_text', '2nd_text', '3rd_text'], axis=1, inplace=True)
        return query_df, class_df

