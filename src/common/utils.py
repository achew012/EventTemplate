import jsonlines
from collections import OrderedDict
import re
import json
import os
import ipdb
from typing import List, Dict, Any, Tuple
from collections import Counter
import torch



def to_jsonl(filename: str, file_obj):
    resultfile = open(filename, "wb")
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


def read_json(jsonfile):
    with open(jsonfile, "rb") as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object


def write_json(filename, file_object):
    with open(filename, "w") as file:
        file.write(json.dumps(file_object))

def cos_sim(input1, input2, eps=1e-8):
    '''
    argument: out: graph embedding
    return: similarity matrix of different graph embedding
    '''
    input1_norm = input1 / (input1.norm(dim=1).unsqueeze(dim=-1) + eps)
    input2_norm = input2 / (input2.norm(dim=1).unsqueeze(dim=-1) + eps)
    sim_mat = torch.mm(input1_norm, input2_norm.T)
    
    return sim_mat

def generate_indicator_matrix(batch):
    '''
    argument: batch: batch data, use batch.y to generate similarity indicator matrix
    return: indicator matrix by shape (N, N), while N is batchsize
    '''
    groundtruth_matrix = batch.y.squeeze(dim=-1).repeat(batch.y.shape[0], 1)
    indicator_matrix = ((groundtruth_matrix == groundtruth_matrix.T) + 0)
    
    return indicator_matrix

def ground_truth_count(batch):
    
    res = 0
    element_dict = Counter(batch.y.squeeze(dim=-1).cpu().numpy())
    for value in element_dict.values():
        res += value ** 2
    
    return res
