import pandas as pd
import torch
import torch.nn as nn
from preprocess import preprocess_class

submission = pd.read_csv('../input/답안 작성용 파일.csv', encoding='CP949')
WC_submission = pd.read_csv('../submission/submission_WC_forensemble.csv')
MC_submission = pd.read_csv('../submission/submission_choice10_forensemble.csv')
class_df = preprocess_class()

WC_logits = []
MC_logits = []
for i in range(len(WC_submission)):
    logit = WC_submission['logits'][i]
    logit = [float(item) for item in logit[1:-1].split(',')]
    WC_logits.append(logit)
for i in range(len(MC_submission)):
    logit = MC_submission['logits'][i]
    logit = [float(item) for item in logit[1:-1].split(',')]
    MC_logits.append(logit)
WC_logits = torch.tensor(WC_logits)
MC_logits = torch.tensor(MC_logits)
softmax = nn.Softmax(dim=-1)
WC_prob = softmax(WC_logits)
MC_prob = softmax(MC_logits)
final_prob = (WC_prob/2) + (MC_prob/2)
final_prob = final_prob.numpy()
preds = final_prob.argmax(-1)
first = []
second = []
third = []
for pred in preds:
    a = list(class_df[class_df['class_num']==pred]['1st'])[0]
    b = list(class_df[class_df['class_num']==pred]['2nd'])[0]
    c = list(class_df[class_df['class_num']==pred]['3rd'])[0]
    first.append(a)
    second.append(b)
    third.append(c)
submission['digit_1'] = first
submission['digit_2'] = second
submission['digit_3'] = third
submission.to_csv(f'../submission/submission_ensemble.csv', index=False)

