import os
import sys
import json
import pandas as pd
import ast

# Post-processing for ACC.

def opposite_acc(component_type):

    if component_type == "Premise":
        return "Claim"
    elif component_type == "Claim":
        return "Premise"
    elif component_type == "MajorClaim":
        return "Claim"

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

# Post-processing for ARI.

def get_ac_count(x):

    return len(ast.literal_eval(x.AC_types))

def get_all_possible_pairs(x):

    pairs_l = []

    ac_count = x.AC_count
    ar_pairs = ast.literal_eval(x.AR_pairs)
    
    for i in range(ac_count):
        for j in range(ac_count):
            if i != j:
                pair = (i, j)
                if pair in ar_pairs:
                    pairs_l.append([i, j, "Rel"])
                else:
                    pairs_l.append([i, j, "N-Rel"])

    return pairs_l

def process_preds(test_preds, LI_grounds_l):

    if len(test_preds) != len(LI_grounds_l):
        print("error in preds length, equalizing lengths ... ")

    preds_pairs_l = []

    for l_1, l_2 in zip(test_preds, LI_grounds_l):
        para_list = []
        for i, j, _ in l_2:
            pair = [i, j]
            if pair in l_1:
                para_list.append([i, j, "Rel"])
            else:
                para_list.append([i, j, "N-Rel"])

        preds_pairs_l.append(para_list)

    return preds_pairs_l

def post_process_ari(results):

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))
    data_directory = os.path.join(parent_directory, "data")
    
    pe_df = pd.read_csv(os.path.join(data_directory, "PE_data.csv"))
    df_split = pd.read_csv(os.path.join(data_directory, "train-test-split.csv"), sep=";") 
    pe_df['split'] = pe_df['essay_id'].map(df_split['SET'])
    pe_df["AC_count"] = pe_df.apply(lambda x: get_ac_count(x), axis=1)
    pe_df["LI_grounds"] = pe_df.apply(lambda x: get_all_possible_pairs(x), axis=1)

    preds = results["predictions"]
    preds = [x["content"] for x in preds]
    preds = [json.loads(x)["list_argument_relations"] for x in preds]    


    LI_grounds_l = list(pe_df[pe_df.split == "TEST"].reset_index().LI_grounds)
    LI_preds_l = process_preds(preds, LI_grounds_l)

    LI_grounds = [item for row in LI_grounds_l for item in row]
    LI_preds = [item for row in LI_preds_l for item in row]
        
    task_grounds = [x[2] for x in LI_grounds]
    task_preds = [x[2] for x in LI_preds]    

    return task_grounds, task_preds   


# Post-processing for ARC

def opposite_arc(relation_type):
    
    if relation_type == "Support":
        return "Attack"
    else:
        return "Support"

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