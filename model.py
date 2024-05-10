import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DistMult(nn.Module):
    def __init__(self, numEntity, numRelation, embedding_dim):
        """
        gamma: refer to the N3 regularization hyperparameter
        """
        super(DistMult, self).__init__()
        self.nentity = numEntity
        self.numRelation = numRelation
        self.embedding_dim = embedding_dim
        
        self.entity_embeddings = torch.nn.Embedding(numEntity, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(numRelation, embedding_dim)
        
        # Xavier uniform initialization for both entity and relation embeddings
        torch.nn.init.xavier_uniform_(self.entity_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, sample):
        positive_samples, negative_samples = sample
        
        # true head, relation and tail
        head_id, relation_id, tail_id = positive_samples[:, 0], positive_samples[:, 1], positive_samples[:, 2]

        # [TODO: ]should we unsqueeze the embedding?
        head_embedding = self.entity_embeddings(head_id).unsqueeze(1)
        relation_embedding = self.relation_embeddings(relation_id).unsqueeze(1)
        tail_embedding = self.entity_embeddings(tail_id).unsqueeze(1)

        fact_score = self.score(head_embedding, relation_embedding, tail_embedding)

        negative_heads, negative_tails = negative_samples["head"], negative_samples["tail"]

        # head prediction forward step
        batch_size, negative_sample_size = negative_heads.size(0), negative_heads.size(1)
        head = self.entity_embeddings(negative_heads.view(-1)).view(batch_size, negative_sample_size, -1)
        head_prediction_score = self.score(head, relation_embedding, tail_embedding)

        # tail prediction forward step
        batch_size, negative_sample_size = negative_tails.size(0), negative_tails.size(1)
        tail = self.entity_embeddings(negative_tails.view(-1)).view(batch_size, negative_sample_size, -1)
        tail_prediction_score = self.score(head_embedding, relation_embedding, tail)

        return fact_score, head_prediction_score, tail_prediction_score
    
    def score(self, head, relation, tail):
        return (head * relation * tail).sum(dim=2)
