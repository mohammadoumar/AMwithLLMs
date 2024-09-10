import os
import ast
import sys
import json
import pandas as pd


# Post-processing for ACC.

def opposite_acc(component_type):

    if component_type == "Claim":
        return "Premise"
    elif component_type == "Premise":
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


#### ARIC post-processing

def process_lists(l):

    l_new = [] 
    for ll in l:
        ll_tmp = []
        for item in ll:
            if item not in ll_tmp:
                ll_tmp.append(item)
        # ll = [list(set(x)) for x in ll]
        l_new.append(ll_tmp)

    return l_new

def clean_preds(nr_acs, current_preds_arg):

    current_preds = current_preds_arg[:]
    
    for pred in current_preds_arg:
        
        if len(pred) != 3:
            current_preds.remove(pred)
        elif pred[0] == pred[1]:
            current_preds.remove(pred)
        elif pred[2] != 'support' and pred[2] != "attack":
            current_preds.remove(pred)
        elif (type(pred[0]) == int and pred[0] > nr_acs) or (type(pred[1]) == int and pred[1] > nr_acs):
            current_preds.remove(pred)
        elif (type(pred[0]) == int and pred[0] <= 0) or (type(pred[1]) == int and pred[1] <= 0):
            current_preds.remove(pred)
        elif type(pred[0]) != int or type(pred[1]) != int:
            current_pred.remove(pred)

    return current_preds

def compute_acs_nr(test_set, DATASET_DIR):

    df = pd.read_pickle(os.path.join(DATASET_DIR, f"""{test_set}_test_df.pkl"""))
    nr_acs_l = [len(element) for element in list(df.acs_list)]

    return nr_acs_l

def get_all_relations(nr_acs_l, grounds_arg, preds_arg):

    grounds_arg = process_lists(grounds_arg)
    preds_arg = process_lists(preds_arg)
    
    final_grounds = []
    final_preds = []

    for idx, nr_acs in enumerate(nr_acs_l):
        
        current_grounds = grounds_arg[idx]
        current_grounds_st = [[x[0], x[1]] for x in current_grounds]
        current_preds = preds_arg[idx]
        current_preds = clean_preds(nr_acs, current_preds)
        current_preds_st = [[x[0], x[1]] for x in current_preds]       

        
        for i in range(1, nr_acs+1):
            for j in range(1, nr_acs+1):
                
                if i != j:
                    
                    st = [i, j]
                    
                    if st not in current_grounds_st:
                        current_grounds.append([st[0], st[1], "None"])

                    if st not in current_preds_st:
                        current_preds.append([st[0], st[1], "None"])

        current_grounds.sort()
        current_preds.sort()
        final_grounds.append(current_grounds)
        final_preds.append(current_preds)

    return final_grounds, final_preds

def post_process_aric(results, nr_acs_l):

    grounds = results["ground_truths"]
    preds = results["predictions"]
    
    grounds = [json.loads(x)["list_argument_relation_types"] for x in grounds]  
    
    preds = [x["content"] for x in preds]    
    preds = [json.loads(x)["list_argument_relation_types"] for x in preds]

    task_grounds, task_preds = get_all_relations(nr_acs_l, grounds, preds)
    

    task_grounds = [item for row in task_grounds for item in row]
    task_preds = [item for row in task_preds for item in row]
    

    final_grounds = [x[2] for x in task_grounds]
    final_preds = [x[2] for x in task_preds]

    return final_grounds, final_preds