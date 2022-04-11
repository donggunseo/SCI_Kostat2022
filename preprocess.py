import pandas as pd
from tqdm import tqdm

## train data와 test data를 읽어와 pandas dataframe형태로 저장
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

# label에 대한 정보가 담겨있는 엑셀 파일을 불러와 dataframe으로 저장
def preprocess_class():
    class_df = pd.read_excel('../input/한국표준산업분류(10차)_국문.xlsx')
    new_header = class_df.iloc[0]
    class_df = class_df[1:]
    class_df.columns=new_header
    class_df.drop([1], axis=0, inplace=True)
    class_df.columns = ['1st', '1st_text', '2nd', '2nd_text', '3rd', '3rd_text', '4th', '4th_text', '5th', '5th_text']
    columns = list(class_df.columns)
    # 엑셀 파일에서 사용하지 않는 분류 단위는 버리고 공백 부분을 해당 분류 값으로 채워 공백을 지움
    for column in columns:  
        class_df[column] = class_df[column].fillna(method = 'ffill')
    class_df.drop(['4th', '4th_text', '5th', '5th_text'], axis =1, inplace=True)
    class_df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    class_df['class_num'] = [i for i in range(len(class_df))]
    # train data의 산업 분류 코드와 format을 맞춰주기 위해 앞자리에 0이 붙은 경우 이를 지우고 저장
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

# text_obj, text_mthd, text_deal을 연결하여 하나의 query 문장으로 저장
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

## dataset 구축전 필요없는 column을 버리고 query dataframe에 각 데이터의 산업 분류 레이블을 번호로 추가해줌
## class_df의 경우 훈련과정에서는 필요없으나 예측 결과를 얻는 과정에서 산업코드를 가져오기 위해 return에 넣어줌
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

