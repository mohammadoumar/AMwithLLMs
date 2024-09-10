import os
import ast
import sys
import json
import datasets
import pandas as pd

from datasets import load_dataset

cdcp_dataset = load_dataset("DFKI-SLT/cdcp", trust_remote_code=True)

nr_acs_l = []

for sample in cdcp_dataset['test']:
    nr_acs_l.append(len(sample['propositions']['label']))

# Post-processing for ACC.

def opposite_acc(component_type):

    if component_type == "fact":
        return "value"
    elif component_type == "value":
        return "policy"
    elif component_type == "policy":
        return "value"
    elif component_type == "testimony":
        return "fact"
    elif component_type == "reference":
        return "policy"

def harmonize_preds_acc(grounds, preds):

    l1, l2 = len(preds), len(grounds)
    if l1 < l2:
        diff = l2 - l1
        preds = preds + [opposite_acc(x) for x in grounds[l1:]]
    else:
        preds = preds[:l2]
        
    return preds 

def post_process_acc(results):

    grounds = results["ground_truths"]
    preds = results["predictions"]
    
    grounds = [json.loads(x)["component_types"] for x in grounds]  
    
    preds = [x["content"] for x in preds]    
    preds = [json.loads(x)["component_types"] for x in preds]
    
    for i,(x,y) in enumerate(zip(grounds, preds)):
    
        if len(x) != len(y):
            
            preds[i] = harmonize_preds_acc(x, y)
            
    task_preds = [item for row in preds for item in row]
    task_grounds = [item for row in grounds for item in row]

    return task_grounds, task_preds


### ARI post-processing

def process_grounds_ari(grounds_l, nr_acs_l):

    pairs_l = []
    
    for idx, ac_count in enumerate(nr_acs_l):
        for i in range(ac_count):
            for j in range(ac_count):

                if i != j:
                    pair = [i, j]
                    if pair in grounds_l[idx]:
                        pairs_l.append([i, j, "Rel"])
                    else:
                        pairs_l.append([i, j, "N-Rel"])

    return pairs_l


def process_preds_ari(preds_l, nr_acs_l):

    pairs_l = []
    
    for idx, ac_count in enumerate(nr_acs_l):
        for i in range(ac_count):
            for j in range(ac_count):

                if i != j:
                    pair = [i, j]
                    if pair in preds_l[idx]:
                        pairs_l.append([i, j, "Rel"])
                    else:
                        pairs_l.append([i, j, "N-Rel"])

    return pairs_l

def post_process_ari(results):

    grounds = results["ground_truths"]
    preds = results["predictions"]

    grounds = [json.loads(x)["list_argument_relations"] for x in grounds]

    preds = [x["content"] for x in preds]
    preds = [json.loads(x)["list_argument_relations"] for x in preds]

    final_grounds = process_grounds_ari(grounds, nr_acs_l)
    final_preds = process_preds_ari(preds, nr_acs_l)

    task_grounds = [x[2] for x in final_grounds]
    task_preds = [x[2] for x in final_preds]

    return task_grounds, task_preds


#### ARC post-processing

def opposite_arc(component_type):

    if component_type == "reason":
        return "evidence"
    elif component_type == "evidence":
        return "reason"

def harmonize_preds_arc(grounds, preds):

    l1, l2 = len(preds), len(grounds)
    if l1 < l2:
        diff = l2 - l1
        preds = preds + [opposite_arc(x) for x in grounds[l1:]]
    else:
        preds = preds[:l2]
        
    return preds 

def post_process_arc(results):

    grounds = results["ground_truths"]
    preds = results["predictions"]
    
    grounds = [json.loads(x)["relation_types"] for x in grounds]  
    
    preds = [x["content"] for x in preds]    
    preds = [json.loads(x)["relation_types"] for x in preds]
    
    for i,(x,y) in enumerate(zip(grounds, preds)):
    
        if len(x) != len(y):
            
            preds[i] = harmonize_preds_arc(x, y)
            
    task_preds = [item for row in preds for item in row]
    task_grounds = [item for row in grounds for item in row]

    return task_grounds, task_preds