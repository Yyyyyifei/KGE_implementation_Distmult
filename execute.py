import json
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model import DistMult
from dataloader import TrainDataset, TestDataset

# Hyperparameter Definition
batch_count = 500
lr = 1e-4
# lambda for N2 regularization
lambda_reg = 1e-4
negative_samples_per_fact = 2
embedding_dimension = 5
epoch = 1

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--embedding_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)

    return parser.parse_args(args)

def collate_fn_train(data):
    """
    rewrite collate_fn structure because we use dictionary when returning items
    """
    positive_samples, negative_samples_dicts = zip(*data)

    batch_positive_samples = torch.stack(positive_samples)

    batch_negative_samples = {
        "head": torch.stack([neg_samples["head"] for neg_samples in negative_samples_dicts]),
        "tail": torch.stack([neg_samples["tail"] for neg_samples in negative_samples_dicts])
    }

    return [batch_positive_samples, batch_negative_samples]

def collate_fn_test(data):
    positive_sample, negative_samples_dicts = zip(*data)

    batch_positive_samples = torch.stack(positive_sample)

    batch_negative_samples = {
        "head": torch.stack([neg_samples["head"] for neg_samples in negative_samples_dicts]),
        "tail": torch.stack([neg_samples["tail"] for neg_samples in negative_samples_dicts]),
        "head_filter": torch.stack([neg_samples["head_filter"] for neg_samples in negative_samples_dicts]),
        "tail_filter": torch.stack([neg_samples["tail_filter"] for neg_samples in negative_samples_dicts])
    }

    return [batch_positive_samples, batch_negative_samples]

def extract_triple(file_path, numEntity, numRelation, entity_dict, relation_dict):
    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            if head not in entity_dict:
                entity_dict[head] = numEntity
                numEntity += 1
            if tail not in entity_dict:
                entity_dict[tail] = numEntity
                numEntity += 1
            if relation not in relation_dict:
                relation_dict[relation] = numRelation
                numRelation += 1
    
    return numEntity, numRelation, entity_dict, relation_dict

def extract_element(data_path):
    numEntity = 0
    numRelation = 0

    entity_dict = {}
    relation_dict = {}

    train_file_path = os.path.join(data_path, "train.txt")
    test_file_path = os.path.join(data_path, "test.txt")
    valid_file_path = os.path.join(data_path, "valid.txt")

    numEntity, numRelation, entity_dict, relation_dict = extract_triple(train_file_path, numEntity, numRelation, entity_dict, relation_dict)
    numEntity, numRelation, entity_dict, relation_dict = extract_triple(test_file_path, numEntity, numRelation, entity_dict, relation_dict)
    numEntity, numRelation, entity_dict, relation_dict = extract_triple(valid_file_path, numEntity, numRelation, entity_dict, relation_dict)

    with open(os.path.join(data_path, "entity_mapping.json"), "w") as f:
        json.dump(entity_dict, f)
    
    with open(os.path.join(data_path, "relation_mapping.json"), "w") as f:
        json.dump(relation_dict, f)

    return numEntity, numRelation, entity_dict, relation_dict

def extract_triples(file_path, entity_dict, relation_dict):
    all_triples = []
    with open(file_path, "r") as f:
        for line in f:
            head, relation, tail = line.strip().split("\t") 
            all_triples.append((
                            entity_dict[head], 
                            relation_dict[relation], 
                            entity_dict[tail]
            ))

    return all_triples

def process_data(data_path):
    # extract entity and relation mapping to id
    entity_mapping_file = os.path.join(data_path, "entity_mapping.json")
    relation_mapping_file = os.path.join(data_path, "relation_mapping.json")
    if os.path.exists(entity_mapping_file) and relation_mapping_file:
        with open(entity_mapping_file, "r") as f:
            entity_dict = json.load(f)
            numEntity = len(entity_dict)
        with open(relation_mapping_file, "r") as f:
            relation_dict = json.load(f)
            numRelation = len(relation_dict)
    else:
        numEntity, numRelation, entity_dict, relation_dict = extract_element(data_path)
    
    train_file_path = os.path.join(data_path, "train.txt")
    test_file_path = os.path.join(data_path, "test.txt")
    valid_file_path = os.path.join(data_path, "valid.txt")

    train_triples = extract_triples(train_file_path, entity_dict, relation_dict)
    test_triples = extract_triples(test_file_path, entity_dict, relation_dict)
    valid_triples = extract_triples(valid_file_path, entity_dict, relation_dict)
    all_triples = train_triples + test_triples + valid_triples

    train_dataset = DataLoader(
        TrainDataset(train_triples, numEntity, negative_sample_size=negative_samples_per_fact),
        shuffle=True,
        collate_fn=collate_fn_train,
        batch_size= (len(train_triples) // batch_count) + 1
    )

    test_dataset = DataLoader(
        TestDataset(test_triples, all_triples, numEntity),
        shuffle=True,
        collate_fn=collate_fn_test,
        batch_size=2
    )

    return train_dataset, test_dataset, numEntity, numRelation, len(all_triples)

def multiclass_NLL(fact_score, negative_score):
    return (-fact_score + torch.log((torch.exp(negative_score).sum()))).sum()

def train_step(model, train_dataset, epoch, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        optimizer.zero_grad()
        
        for i, (positive_sample, negative_samples) in enumerate(train_dataset):
            positive_sample.to(device)
            negative_samples["head"].to(device)
            negative_samples["tail"].to(device)
            fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples))

            regularization = lambda_reg * (
                model.entity_embeddings.weight.norm(p = 2) + 
                model.relation_embeddings.weight.norm(p = 2)
            )

            loss = (multiclass_NLL(fact_score, head_prediction_score) + multiclass_NLL(fact_score, tail_prediction_score)) / batch_size + regularization

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        print(f"On epoch {e}, loss is {loss}")

def calculate_ranking(positive_sample, negative_samples, fact_score, prediction_score, filter, batch_size, is_head):
    pos = 0 if is_head else 2 

    # replace scores of filtered entities with min_int
    min_int_value = torch.iinfo(torch.int32).min
    prediction_score = torch.where(filter, prediction_score, torch.IntTensor([min_int_value]))
    all_scores = torch.cat((fact_score, prediction_score), dim=1)

    fact_with_negative = torch.cat((positive_sample[:, pos].unsqueeze(1), negative_samples), dim=1)

    sorted_indicies = torch.argsort(all_scores, descending=True)
    sorted_heads = torch.gather(fact_with_negative, 1, sorted_indicies)

    rankings = []
    for i in range(batch_size):
        index = (sorted_heads[i, :] == positive_sample[i][pos]).nonzero()
        # ranking should only have two occurence
        # the latter one has score min_int
        print(index)
        rankings.append(index.item() + 1)
    
    return torch.IntTensor(rankings)

def test_step(model, test_dataset, device):
    model.eval()

    num_samples = 0
    MRR = 0
    HitAt_1 = 0
    HitAt_3 = 0
    HitAt_5 = 0

    with torch.no_grad():
        epoch = 0
        for positive_sample, negative_samples in test_dataset:
            positive_sample.to(device)

            negative_samples["head"].to(device)
            negative_samples["tail"].to(device)
        
            fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples))

            batch_size = fact_score.size(0)

            head_ranking = calculate_ranking(
                positive_sample, 
                negative_samples["head"], 
                fact_score, 
                head_prediction_score, 
                negative_samples["head_filter"], 
                batch_size,
                True
            )

            tail_ranking = calculate_ranking(
                positive_sample, 
                negative_samples["tail"], 
                fact_score, 
                tail_prediction_score, 
                negative_samples["tail_filter"], 
                batch_size,
                False
            )

            print(head_ranking)
            print(tail_ranking)

            # can do head / tail prediction evaluation separately
            # here we merge them for convenience
            batch_mrr = torch.sum(1.0 / head_ranking) + torch.sum(1.0 / tail_ranking)

            batch_hitAt_1 = torch.sum(torch.where(head_ranking <= 1, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))
            batch_hitAt_3 = torch.sum(torch.where(head_ranking <= 3, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))
            batch_hitAt_5 = torch.sum(torch.where(head_ranking <= 5, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))

            MRR += batch_mrr
            HitAt_1 += batch_hitAt_1
            HitAt_3 += batch_hitAt_3
            HitAt_5 += batch_hitAt_5
            num_samples += batch_size

            if epoch < 10:
                epoch += 1
            else:
                break
    
    if num_samples != 0:
        print(f"MRR is {MRR / num_samples}")
        print(f"HitAt_1 is {HitAt_1 / num_samples}")
        print(f"HitAt_3 is {HitAt_3 / num_samples}")
        print(f"HitAt_5 is {HitAt_5 / num_samples}")


if __name__ == "__main__":
    train_dataset, test_dataset, numEntity, numRelation, numSamples = process_data(data_path=os.path.join(os.getcwd(), "data"))

    if torch.cuda.is_available():
        print("CUDA available, training on GPU ...")
    else:
        print("CUDA not available, training on CPU ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistMult(numEntity, numRelation, embedding_dimension)
    model.to(device)

    args = parse_args(args)

    train_step(model, train_dataset, 50, device)
    test_step(model, test_dataset, device)