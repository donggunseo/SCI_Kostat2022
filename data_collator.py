import torch
from typing import Optional, Union
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from itertools import chain

@dataclass
class DataCollatorForMultipleChoice:
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    do_train: bool = True

    def __call__(self, features):
        if self.do_train:
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if self.do_train:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch