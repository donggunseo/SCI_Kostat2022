# SCI_Kostat2022
2022 통계데이터 인공지능 활용대회 GIST SCI LAB
## Team Members
|서동건(Team Leader)|김주영|김주연|
| :---: | :---: | :---: |
|<a href="https://github.com/donggunseo" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/43330160?v=4" width="80%" height="80%">| <a href="https://github.com/jspirit01" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/55519927?v=4" width="80%" height="80%"> | <a href="https://github.com/superjuyeon" height="5" width="10" target="_blank"><img src="https://avatars.githubusercontent.com/u/66545973?v=4" width="80%" height="80%"> |
|GIST EECS Bachelor student|GIST IIT Combined MS/PhD student|GIST IIT MS student|
|Modeling, Tuning|EDA, Data cleaning|EDA, ML-based approach|
## 😀Directory 
```bash
kostat
├── SCI_Kostat2022
│   ├── README.md
│   ├── create_kfold.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── train_MD.py
│   ├── train_WC.py
│   └── utils.py
└── input
    ├── 1. 실습용자료.txt
    ├── 2. 모델개발용자료.txt
    ├── 답안 작성용 파일.csv
    └── 한국표준산업분류(10차)_국문.xlsx
```
## 😀Environment setting
Default python version == 3.9.10
```
pip install -r requirements.txt
```
## 😀Train model(WC)
Using Default AutoModelForSequenceClassification from Huggingface transformers
```
python train_WC.py --kfold 5
```
## 😀Train model(MD)
Using Custom model which Multi-Dropout is applied (model is implemented in model.py)
```
python train_MD.py -=kfold 5
```

<br>

## 😀Inference
Choose best CV checkpoint model for each fold \
All hyperparameters used to get results below are described in code
### CV accuracy for each model 

|  | WC | MD |
| --- | --- | --- |
| fold0 | <u>93.032</u> | 92.963 |
| fold1 | <u>92.965</u> | 92.954 |
| fold2 | 93.015 | <u>93.033</u> |
| fold3 | 92.954 | <u>92.991</u> |
| fold4 | 92.918 | <u>92.954</u> |

```python
## you can edit this checkpoint list depended on your result
model_checkpoint = [f'../best_model/roberta_large_WC_fold{fold}' for fold in range(0,2)]
model_checkpoint1 = [f'../best_model/roberta_large_WC_MD_fold{fold}' for fold in range(2,5)]
model_checkpoint.extend(model_checkpoint1)
inference(model_checkpoint) 
```
```
python inference.py
```
