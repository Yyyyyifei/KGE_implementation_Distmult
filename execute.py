import argparse
import logging
import json
import os
import matplotlib.pyplot as plt
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
    parser.add_argument('-d', '--embedding_dim', default=300, type=int)
    parser.add_argument('-e', '--epoch', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-l', '--lambda_reg', default=1e-4, type=float)
    parser.add_argument('-n', '--negative_sample_size', default=256, type=int)
    parser.add_argument('-p', '--model_path', default="", type=str)
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--loadprev', action='store_true', default=False)

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
        batch_size=4
    )

    valid_dataset = DataLoader(
        TestDataset(valid_triples, all_triples, numEntity),
        shuffle=True,
        collate_fn=collate_fn_test,
        batch_size=4
    )

    return train_dataset, test_dataset, valid_dataset, numEntity, numRelation, len(all_triples)

def multiclass_NLL(fact_score, negative_score):
    combined_score = torch.cat([fact_score, negative_score], dim=1)
    return -fact_score + torch.logsumexp(combined_score, dim=1, keepdim=True)

def multiclass_NLL_2(fact_score, negative_score):
    return (-fact_score + torch.log(torch.exp(negative_score).sum())).sum()

def train_step(model, train_dataset, valid_dataset, epoch, device, lr, lambda_reg):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_losses = []
    all_mrr = []

    for e in range(epoch):
        model.train()
        optimizer.zero_grad()
        
        total_epoch_loss = 0
        num_batches = 0

        mrr_prev = 0
        patience = 2

        for i, (positive_sample, negative_samples) in enumerate(train_dataset):
            positive_sample.to(device)
            negative_samples["head"].to(device)
            negative_samples["tail"].to(device)
            fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples), device)

            batch_size = fact_score.size(0)

            regularization = lambda_reg * (
                model.entity_embeddings.weight.norm(p = 3) + 
                model.relation_embeddings.weight.norm(p = 3)
            )

            batch_loss = (multiclass_NLL(fact_score, head_prediction_score) + multiclass_NLL(fact_score, tail_prediction_score)).mean()
            
            total_epoch_loss += batch_loss
            num_batches += 1

            loss = batch_loss + regularization

            if torch.isnan(loss).any():
                logging.warning(f"NaN loss detected at epoch {e}")
                break

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = total_epoch_loss / num_batches

        if torch.isnan(epoch_loss).any():
            logging.warning(f"NaN loss detected at epoch {e}")

        logging.info(f"On epoch {e}, loss is {epoch_loss}")
        all_losses.append(epoch_loss)

        if (e + 1) % 5 == 0:
            validation_mrr = test_step(model, valid_dataset, device)
            all_mrr.append(validation_mrr)

            logging.info(f"Validation MRR on epoch {e} is {validation_mrr}")

            checkpoint = os.path.join("/h/224/yfsun/KGE_implementation_Distmult/models", f"model_{e+1}.pth")
            save_checkpoint({
                "epoch" : epoch + 1,
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict()
            }, checkpoint)

            if validation_mrr < mrr_prev:
                patience -= 1
            else:
                mrr_prev = validation_mrr

        if patience <= 0:
            break

    return all_losses, all_mrr

def validation_step(model, valid_dataset, device, lambda_reg):
    model.eval()
    
    total_epoch_loss = 0
    num_batches = 0

    for _, (positive_sample, negative_samples) in enumerate(valid_dataset):
        positive_sample.to(device)
        negative_samples["head"].to(device)
        negative_samples["tail"].to(device)
        fact_score, head_prediction_score, tail_prediction_score = model.forward((positive_sample, negative_samples), device)

        regularization = lambda_reg * (
            model.entity_embeddings.weight.norm(p = 3) + 
            model.relation_embeddings.weight.norm(p = 3)
        )

        batch_loss = (multiclass_NLL(fact_score, head_prediction_score) + multiclass_NLL(fact_score, tail_prediction_score)).mean()
        loss = batch_loss + regularization

        num_batches += 1
        total_epoch_loss += loss
    
    return total_epoch_loss / num_batches

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
    HitAt_5 = 0
    HitAt_10 = 0

    with torch.no_grad():
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
    
    if num_samples != 0:
        logging.info(f"MRR is {MRR / (num_samples * 2)}")
        logging.info(f"HitAt_1 is {HitAt_1 / num_samples}")
        logging.info(f"HitAt_5 is {HitAt_5 / num_samples}")
        logging.info(f"HitAt_10 is {HitAt_10 / num_samples}\n")

        return MRR / (num_samples * 2)

    return None

def plot_losses(epoch, metrics, step=1, figure_metric=''):
    x = range(step, epoch + 1, step)

    fig, ax = plt.subplots()

    ax.plot(x, metrics, linestyle='-', color='b')

    ax.set_xlabel('epoch')
    ax.set_ylabel(f'{figure_metric}')

    ax.set_title(f"Epoch VS {figure_metric}")

    fig.savefig(os.path.join(os.getcwd(), f'{figure_metric}.png'), dpi=300)

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)

    # args = parse_args()
    
    # # use batch count instead of batch_size to control batches
    # batch_count = args.batch_count
    # lr = args.learning_rate
    # # lambda for N2 regularization
    # lambda_reg = args.lambda_reg
    # negative_samples_per_fact = args.negative_sample_size
    # embedding_dimension = args.embedding_dim
    # epoch = args.epoch

    # train_dataset, test_dataset, valid_dataset, numEntity, numRelation, numSamples = process_data("/h/224/yfsun/KGE_implementation_Distmult/data", negative_samples_per_fact, batch_count)

    # if torch.cuda.is_available():
    #     logging.info("CUDA available, training on GPU ...")
    # else:
    #     logging.warning("CUDA not available, training on CPU ...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.loadprev:
    #     # load model for testing
    #     model = DistMult(numEntity, numRelation, embedding_dimension)

    #     checkpoint = torch.load(args.path)
    #     model.load_state_dict(checkpoint["state_dict"], strict=False)

    #     model.to(device)
    #     test_step(model, test_dataset, device)

    # else:
    #     model = DistMult(numEntity, numRelation, embedding_dimension)
    #     model.to(device)

    #     logging.info(f"Running training on the following parameters: \n \
    #         batch_count : {batch_count} \n \
    #         lr : {lr} \n \
    #         lambda_reg : {lambda_reg} \n \
    #         negative_samples_per_fact : {negative_samples_per_fact}\n \
    #         embedding_dimension : {embedding_dimension}\n \
    #         epoch = {epoch}"
    #     )

    #     train_losses, validation_mrr = train_step(model, train_dataset, valid_dataset, epoch, device, lr, lambda_reg)

    #     logging.info(train_losses)
    #     logging.info(validation_mrr)

    #     if not args.validation:
    #         test_step(model, test_dataset, device)
    
    epoch = 150

    train_losses = [11.0981, 11.098, 11.0975, 11.0959, 11.0893, 11.0698, 11.0263, 10.9482, 10.8279, 10.6603, 10.443, 10.178, 9.8706, 9.5315, 9.1729, 8.8074, 8.4455, 8.0945, 7.7568, 7.4326, 7.126, 6.8373, 6.5663, 6.316, 6.0843, 5.8698, 5.6766, 5.4933, 5.3272, 5.1725, 5.0291, 4.8952, 4.7718, 4.654, 4.5436, 4.4433, 4.3462, 4.2529, 4.165, 4.0814, 4.0005, 3.9255, 3.8528, 3.7813, 3.7142, 3.6493, 3.5878, 3.5278, 3.4697, 3.4117, 3.3567, 3.3014, 3.2491, 3.198, 3.1475, 3.1002, 3.0508, 3.0043, 2.9606, 2.9141, 2.8695, 2.8273, 2.7846, 2.7456, 2.7051, 2.6626, 2.6277, 2.586, 2.5496, 2.5125, 2.476, 2.4407, 2.4056, 2.3722, 2.3352, 2.3043, 2.2709, 2.2372, 2.205, 2.1747, 2.1418, 2.1116, 2.0806, 2.0526, 2.0212, 1.9924, 1.9628, 1.9356, 1.9073, 1.8778, 1.8487, 1.8227, 1.7956, 1.7694, 1.7454, 1.7168, 1.691, 1.6633, 1.6393, 1.6146, 1.5877, 1.5648, 1.5395, 1.5145, 1.4932, 1.4697, 1.4439, 1.4224, 1.4025, 1.3773, 1.3548, 1.3332, 1.3102, 1.2884, 1.2673, 1.2485, 1.2262, 1.2053, 1.186, 1.1653, 1.1446, 1.1273, 1.1067, 1.0873, 1.07, 1.0492, 1.0301, 1.0143, 0.9973, 0.9777, 0.9611, 0.9425, 0.9279, 0.9111, 0.8923, 0.8774, 0.8616, 0.8463, 0.8307, 0.8151, 0.8008, 0.7829, 0.7704, 0.754, 0.7423, 0.7279, 0.7149, 0.6994, 0.6855, 0.674]
    validation_mrr = [0.1903, 0.2236, 0.2247, 0.2281, 0.233, 0.2385, 0.2457, 0.253, 0.2603, 0.2668, 0.2718, 0.2761, 0.2804, 0.2849, 0.288, 0.2911, 0.2944, 0.2962, 0.2984, 0.3, 0.3028, 0.3049, 0.3057, 0.306, 0.3058, 0.3054, 0.3042, 0.3038, 0.3018, 0.3002]

    plot_losses(epoch, train_losses, 1, "train_losses")
    plot_losses(epoch, validation_mrr, 5, "validation_mrr")