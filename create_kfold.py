from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

## sklearn의 StraitifiedKFold 클래스를 통해 각 폴드별로 클래스 분포가 비슷하도록 나눠주는 함수
## dataframe에 몇 번 폴드에 해당하지는지 column을 추가해 정보를 남겨줌
def create_kfold(df, kfold=5):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    df['kfold']=-1
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['class_num'])):
        df.loc[val_index, 'kfold']= fold
    return df
