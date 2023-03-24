
from dataclasses import dataclass

from transformers import PreTrainedTokenizer
import torch

@dataclass
class MNRCollator:

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8
    max_length: int = 512

    def __call__(self, features):

        longest_topic = max([len(x["input_ids_a"]) for x in features])
        longest_content = max([len(x["input_ids_b"]) for x in features])

        pad_token_id = self.tokenizer.pad_token_id

        input_ids_topic = [
            x["input_ids_a"]
            + [pad_token_id]
            * (min(longest_topic, self.max_length) - len(x["input_ids_a"]))
            for x in features
        ]
        attention_mask_topic = [
            x["attention_mask_a"]
            + [0] * (min(longest_topic, self.max_length) - len(x["attention_mask_a"]))
            for x in features
        ]

        input_ids_content = [
            x["input_ids_b"]
            + [pad_token_id]
            * (min(longest_content, self.max_length) - len(x["input_ids_b"]))
            for x in features
        ]
        attention_mask_content = [
            x["attention_mask_b"]
            + [0]
            * (min(longest_content, self.max_length) - len(x["attention_mask_b"]))
            for x in features
        ]

        return {
            "input_ids_a": torch.tensor(input_ids_topic),
            "attention_mask_a": torch.tensor(attention_mask_topic),
            "input_ids_b": torch.tensor(input_ids_content),
            "attention_mask_b": torch.tensor(attention_mask_content),
        }