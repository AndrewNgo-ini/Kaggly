%%writefile modeling.py

from dataclasses import dataclass

import numpy as np
from transformers import AutoModel, Trainer
from transformers.utils import ModelOutput
import torch

from utils import cos_sim


@dataclass
class SBertOutput(ModelOutput):
    """
    Used for SBert Model Output
    """
    loss: torch.Tensor = None
    pooled_embeddings: torch.Tensor = None
        
        
class SBert(torch.nn.Module):
    """
    Basic SBert wrapper. Gets output embeddings and averages them, taking into account the mask.
    """
    def __init__(self, model_name):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """
        Average the output embeddings using the attention mask 
        to ignore certain tokens.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):

        outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)

        return SBertOutput(
            loss=None, # loss is calculated in `compute_loss`, but needed here as a placeholder
            pooled_embeddings=self.mean_pooling(outputs[0], attention_mask),
        )

# Basically the same as this: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
class MultipleNegativesRankingLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row. 
        This indicates that `a_i` and `b_j` have high similarity 
        when `i==j` and low similarity when `i!=j`.
        """

        similarity_scores = (
            cos_sim(embeddings_a, embeddings_b) * 20.0
        )  # Not too sure why to scale it by 20: https://github.com/UKPLab/sentence-transformers/blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L57
        
        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)