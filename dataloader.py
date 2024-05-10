import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# referenced from https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/codes/dataloader.py#L96
def _get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

    return true_head, true_tail

class TrainDataset(Dataset):
    def __init__(self, triples, numEntities, negative_sample_size=128):
        self.triples = triples
        self.nentity = numEntities
        self.negative_sample_size = negative_sample_size
        self.true_head, self.true_tail = _get_true_head_and_tail(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        positive_sample = self.triples[index]

        negative_samples = {
            "head" : torch.LongTensor(self._random_sampling_corruption(self, positive_sample, True)),
            "tail" : torch.LongTensor(self._random_sampling_corruption(self, positive_sample, False))
        }

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_samples

    @staticmethod
    def _random_sampling_corruption(self, positive_sample, is_head: bool):
        """
        corrupting a fact using random sampling corruption
        """
        head, relation, tail = positive_sample
        all_entities = np.arange(0, self.nentity-1)

        if is_head:
            mask = np.isin(
                all_entities, 
                self.true_head[(relation, tail)], 
                assume_unique=True, 
                invert=True
            )
        else:
            mask = np.isin(
                all_entities, 
                self.true_tail[(head, relation)], 
                assume_unique=True, 
                invert=True
            )
        
        negative_sample_pool = all_entities[mask]

        # [TODO:] add handling for cases where could not generate enough negative sample
        negative_samples = np.random.choice(
            negative_sample_pool, 
            size=min(len(negative_sample_pool), self.negative_sample_size), 
            replace=False
        )
        
        return negative_samples

class TestDataset(Dataset):
    def __init__(self, triples, all_triples, numEntities):
        # numEntities refer to the number of ALL entities across train/test/valid dataset
        self.triples = triples
        self.all_triples = all_triples
        self.nentity = numEntities
        self.true_head, self.true_tail = _get_true_head_and_tail(all_triples)

    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, index):
        positive_sample = self.triples[index]

        head_negative_samples, head_filter = self.generate_all_corruption(self, positive_sample, True)
        tail_negative_samples, tail_filter = self.generate_all_corruption(self, positive_sample, False)

        negative_samples = {
            "head" : torch.LongTensor(head_negative_samples),
            "tail" : torch.LongTensor(tail_negative_samples),
            "head_filter" : torch.BoolTensor(head_filter),
            "tail_filter" : torch.BoolTensor(tail_filter)
        }

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_samples

    @staticmethod
    def generate_all_corruption(self, postive_sample, is_head: bool):
        """
        corrupting a fact using random sampling corruption
        """
        head, relation, tail = postive_sample
        all_entities = list(range(self.nentity))

        # filter all true entities across 
        if is_head:
            all_entities.pop(head)
            print(head in all_entities)
            all_entities = np.array(all_entities)
            
            mask = np.isin(
                all_entities, 
                self.true_head[(relation, tail)], 
                assume_unique=True, 
                invert=True
            )
        else:
            all_entities.pop(tail)
            print(tail in all_entities)
            all_entities = np.array(all_entities)

            mask = np.isin(
                all_entities, 
                self.true_tail[(head, relation)], 
                assume_unique=True, 
                invert=True
            )

        return all_entities, mask

# if __name__ == "__main__":
