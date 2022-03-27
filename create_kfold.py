from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
def create_kfold(df, kfold=5):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    df['kfold']=-1
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['class_num'])):
        df.loc[val_index, 'kfold']= fold
    return df
