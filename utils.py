import torch
import random
import os
import numpy as np

## 랜덤시드를 고정하여 실험과정에서 정확한 대조군을 설정하기 위한 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True