from sklearn.model_selection import StratifiedKFold

def create_kfold(df, k=5):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['kfold']=-1
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['class_num'])):
        df.loc[val_index, 'kfold']= fold
    return df
