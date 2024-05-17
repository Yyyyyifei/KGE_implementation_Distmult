import argparse
import logging
import json
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model import DistMult
from dataloader import TrainDataset, TestDataset

# batch_count = 500
# lr = 1e-4
# # lambda for N2 regularization
# lambda_reg = 1e-4
# negative_samples_per_fact = 128
# embedding_dimension = 500
# epoch = 50

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Distmult Models'
    )
    
    parser.add_argument('-b', '--batch_size', default=-1, type=int)
    parser.add_argument("-bc", '--batch_count', default=500, type=int)
    parser.add_argument('-d', '--embedding_dim', default=500, type=int)
    parser.add_argument('-e', '--epoch', default=50, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-l', '--lambda_reg', default=1e-4, type=float)
    parser.add_argument('-n', '--negative_sample_size', default=100, type=int)
    parser.add_argument('--validation', action='store_true', default=False)

    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    return parser.parse_args(args)

def save_checkpoint(state, filename):
    torch.save(state, filename)

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

def process_data(data_path, negative_samples_per_fact, batch_count):
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
        batch_size=(len(train_triples) // batch_count) + 1
    )

    test_dataset = DataLoader(
        TestDataset(test_triples, all_triples, numEntity),
        shuffle=True,
        collate_fn=collate_fn_test,
        batch_size=2
    )

    valid_dataset = DataLoader(
        TrainDataset(valid_triples, numEntity, negative_sample_size=negative_samples_per_fact),
        shuffle=True,
        collate_fn=collate_fn_train,
        batch_size=(len(train_triples) // batch_count) + 1
    )

    return train_dataset, test_dataset, valid_dataset, numEntity, numRelation, len(all_triples)

def multiclass_NLL(fact_score, negative_score):
    return (-fact_score + torch.log((torch.exp(negative_score).sum()))).sum()

def train_step(model, train_dataset, epoch, device, lr, lambda_reg):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    for e in range(epoch):
        model.train()
        optimizer.zero_grad()
        
        total_epoch_loss = 0
        num_samples = 0

        for i, (positive_sample, negative_samples) in enumerate(train_dataset):
            positive_sample.to(device)
            negative_samples["head"].to(device)
            negative_samples["tail"].to(device)
            fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples), device)

            batch_size = fact_score.size(0)

            regularization = lambda_reg * (
                model.entity_embeddings.weight.norm(p = 2) + 
                model.relation_embeddings.weight.norm(p = 2)
            )

            batch_loss = multiclass_NLL(fact_score, head_prediction_score) + multiclass_NLL(fact_score, tail_prediction_score)
            
            total_epoch_loss += batch_loss
            num_samples += batch_size

            loss = batch_loss / batch_size + regularization

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = total_epoch_loss /  num_samples
        
        logging.info(f"On epoch {e}, loss is {epoch_loss}")

        if epoch - e <= 10:
            checkpoint = os.path.join("/h/224/yfsun/KGE_implementation_Distmult/models", f"model_{e+1}.pth")
            save_checkpoint({
                "epoch" : epoch + 1,
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }, checkpoint)

    return all_losses

def calculate_ranking(positive_sample, negative_samples, fact_score, prediction_score, filter, batch_size, is_head):
    pos = 0 if is_head else 2 

    # replace scores of filtered entities with min_int
    min_int_value = torch.iinfo(torch.int32).min
    prediction_score = torch.where(filter, prediction_score, torch.IntTensor([min_int_value]).to(device))
    all_scores = torch.cat((fact_score, prediction_score), dim=1).to(device)

    fact_with_negative = torch.cat((positive_sample[:, pos].unsqueeze(1), negative_samples), dim=1).to(device)

    sorted_indicies = torch.argsort(all_scores, descending=True).to(device)
    sorted_heads = torch.gather(fact_with_negative, 1, sorted_indicies)

    rankings = []
    for i in range(batch_size):
        index = (sorted_heads[i, :] == positive_sample[i][pos]).nonzero()
        # ranking should only have two occurence
        # the latter one has score min_int
        rankings.append(index.item() + 1)
    
    return torch.IntTensor(rankings)

def test_step(model, test_dataset, device):
    model.eval()

    num_samples = 0
    MRR = 0
    HitAt_1 = 0
    HitAt_3 = 0
    HitAt_10 = 0

    with torch.no_grad():
        epoch = 0
        for positive_sample, negative_samples in test_dataset:
            positive_sample.to(device)

            negative_samples["head"].to(device)
            negative_samples["tail"].to(device)
        
            fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples), device)

            batch_size = fact_score.size(0)

            head_ranking = calculate_ranking(
                positive_sample, 
                negative_samples["head"], 
                fact_score, 
                head_prediction_score, 
                negative_samples["head_filter"].to(device), 
                batch_size,
                True
            )

            tail_ranking = calculate_ranking(
                positive_sample, 
                negative_samples["tail"], 
                fact_score, 
                tail_prediction_score, 
                negative_samples["tail_filter"].to(device), 
                batch_size,
                False
            )

            # can do head / tail prediction evaluation separately
            # here we merge them for convenience
            batch_mrr = torch.sum(1.0 / head_ranking) + torch.sum(1.0 / tail_ranking)

            batch_hitAt_1 = torch.sum(torch.where(head_ranking <= 1, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))
            batch_hitAt_5 = torch.sum(torch.where(head_ranking <= 5, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))
            batch_hitAt_10 = torch.sum(torch.where(head_ranking <= 10, torch.FloatTensor([1.0]), torch.FloatTensor([0.0])))

            MRR += batch_mrr
            HitAt_1 += batch_hitAt_1
            HitAt_5 += batch_hitAt_5
            HitAt_10 += batch_hitAt_10
            num_samples += batch_size

            if epoch < 10:
                epoch += 1
            else:
                break
    
    if num_samples != 0:
        logging.info(f"MRR is {MRR / (num_samples * 2)}")
        logging.info(f"HitAt_1 is {HitAt_1 / num_samples}")
        logging.info(f"HitAt_5 is {HitAt_5 / num_samples}")
        logging.info(f"HitAt_10 is {HitAt_10 / num_samples}")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    args = parse_args()
    
    # use batch count instead of batch_size to control batches
    batch_count = args.batch_count
    lr = args.learning_rate
    # lambda for N2 regularization
    lambda_reg = args.lambda_reg
    negative_samples_per_fact = args.negative_sample_size
    embedding_dimension = args.embedding_dim
    epoch = args.epoch

    train_dataset, test_dataset, valid_dataset, numEntity, numRelation, numSamples = process_data("/h/224/yfsun/KGE_implementation_Distmult/data", negative_samples_per_fact, batch_count)

    if torch.cuda.is_available():
        logging.info("CUDA available, training on GPU ...")
    else:
        logging.warning("CUDA not available, training on CPU ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DistMult(numEntity, numRelation, embedding_dimension)
    model.to(device)

    logging.info(f"Running training on the following parameters: \n \
        batch_count :  {batch_count} \n \
        lr : {lr} \n \
        lambda_reg : {lambda_reg} \n \
        negative_samples_per_fact : {negative_samples_per_fact}\n \
        embedding_dimension : {embedding_dimension}\n \
        epoch = {epoch}"
    )

    all_losses = train_step(model, train_dataset, epoch, device, lr, lambda_reg)

    if args.validation:
        test_step(model, valid_dataset, device)
    else:
        test_step(model, test_dataset, device)