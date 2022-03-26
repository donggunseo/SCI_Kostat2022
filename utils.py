import torch
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def post_processing(predictions):
    print(predictions.shape) ##(2000000,2)
    softmax = torch.nn.Softmax(dim=-1)
    predictions = torch.tensor(predictions)
    pred_score = softmax(predictions)
    pred_score = pred_score.view(-1, 10, 2)
    pred_score = pred_score.numpy()
    pred_score = pred_score[:,:,1]
    preds = np.argmax(pred_score, axis=-1)
    return preds
