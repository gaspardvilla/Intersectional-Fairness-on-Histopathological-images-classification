from helpers.pareto_fairness import compute_pareto_metrics
from lightning import seed_everything
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random


loss_fct = nn.CrossEntropyLoss()
metrics_to_test = ['MMPF', 'MMPF_5', 'MMPF_10', 'MMPF_size', 'MMPF_size_2', 'MMPF_adapted', 'MMPF_adapted_2']
attributes = ['att_1', 'att_2']

#### Create a fair and unfair scenario
def build_scenario(data_df : pd.DataFrame, seed : int):
    # Initialization
    seed_everything(seed)
    df_f = data_df.copy()
    df_u = data_df.copy()
    acc_pos = 0.9
    acc_neg = 0.5
    sigma = 0.1
    
    
    # Fair scenario
    # Create 6x6 fair preds ratio
    fair_ratio = np.random.normal(acc_pos, sigma, 36).reshape((6, 6))
    fair_ratio[fair_ratio >= 1] = 0.99
    for att_1 in range_atts[0]:
        for att_2 in range_atts[1]:
            
            # conditon subdf
            cond = (df_f.att_1 == att_1) & (df_f.att_2 == att_2)
            sub_df = df_f[cond].copy()
            n_sub = len(sub_df)
            
            # Build the preds
            n_good = int(n_sub * fair_ratio[att_1][att_2] + 0.5)
            sub_preds = np.ones(n_sub, int)
            sub_preds[:n_good] = 0
            random.shuffle(sub_preds)
            sub_preds = abs(sub_preds - np.array(df_f[cond].label))
            df_f.loc[cond, 'pred'] = sub_preds
            
            # Build the raw preds
            sub_preds_raw = []
            poss = np.random.uniform(0.5, 1, n_sub)
            negs = np.random.uniform(0, 0.5, n_sub)
            for i, p in enumerate(sub_preds):
                pos = poss[i]
                neg = negs[i]
                raw = [0, 0]
                raw[int(p)] = pos
                raw[int(1-p)] = neg
                sub_preds_raw += [raw]
            df_f.loc[cond, 'pred_raw'] = pd.Series(sub_preds_raw)
            
            
    # Unfair scenario
    # Create 6x6 unfair preds ratio
    unfair_ratio1 = np.random.normal(acc_pos, sigma, 18)
    unfair_ratio2 = np.random.normal(acc_neg, sigma, 18)
    unfair_ratio1[unfair_ratio1 >= 1] = 0.99
    unfair_ratio2[unfair_ratio2 >= 1] = 0.99
    unfair_ratio = list(np.concatenate((unfair_ratio1, unfair_ratio2)))
    random.shuffle(unfair_ratio)
    unfair_ratio = np.array(unfair_ratio).reshape((6,6))
    for att_1 in range_atts[0]:
        for att_2 in range_atts[1]:
            
            # conditon subdf
            cond = (df_u.att_1 == att_1) & (df_u.att_2 == att_2)
            sub_df = df_u[cond].copy()
            n_sub = len(sub_df)
            
            # Build the preds
            n_good = int(n_sub * unfair_ratio[att_1][att_2] + 0.5)
            sub_preds = np.ones(n_sub, int)
            sub_preds[:n_good] = 0
            random.shuffle(sub_preds)
            sub_preds = abs(sub_preds - np.array(df_f[cond].label))
            df_u.loc[cond, 'pred'] = sub_preds
            
            # Build the raw preds
            sub_preds_raw = []
            poss = np.random.uniform(0.5, 1, n_sub)
            negs = np.random.uniform(0, 0.5, n_sub)
            for i, p in enumerate(sub_preds):
                pos = poss[i]
                neg = negs[i]
                raw = [0, 0]
                raw[int(p)] = pos
                raw[int(1-p)] = neg
                sub_preds_raw += [raw]
            df_u.loc[cond, 'pred_raw'] = pd.Series(sub_preds_raw)
            
    # reset indices
    df_f.reset_index(inplace = True, drop = True)
    df_u.reset_index(inplace = True, drop = True)
    df_f.reset_index(inplace = True, drop = False)
    df_u.reset_index(inplace = True, drop = False)
    df_f.rename(columns = {'index' : 'subj'}, inplace = True)
    df_u.rename(columns = {'index' : 'subj'}, inplace = True)
    
    # Return fair and unfair
    return df_f, df_u, fair_ratio, unfair_ratio


def check_scenario(df_f, df_u, seed):
    # Initialization
    seed_everything(seed)
    success = {'MMPF' : [],
            'MMPF_5' : [], 'MMPF_10' : [],
            'MMPF_size' : [], 'MMPF_size_2' : [],
            'MMPF_adapted' : [], 'MMPF_adapted_2' : []}
    
    # Extract the subjects and get the 20% test set
    test_subj_list = []
    
    # Get the values of unique subgroups prensent in the data set
    unique_subgroups = np.unique(df_f[attributes].values, axis = 0)

    # Loop on all the unique subgroups in the references
    for subgroup in unique_subgroups:
        
        # Extract the references that are in this subgroup only
        cond = df_f[attributes[0]] == subgroup[0]
        for idx, att in enumerate(attributes[1:]):
            cond = cond & (df_f[att] == subgroup[idx+1])
        subjects = list(df_f[cond].subj)
        
        # Shuffle and split it between train / validation / test sets
        random.shuffle(subjects)
        if len(subjects) == 1: pass
        elif len(subjects) == 2: pass
        elif len(subjects) == 3: test_subj_list += [subjects[2]]
        elif len(subjects) == 4: test_subj_list += [subjects[3]]
        else:
            sub_n = int(0.2 * len(subjects))
            test_subj_list += subjects[: sub_n]

    # Get the results of the data set only
    test_results_fair = df_f[df_f.subj.isin(test_subj_list)]
    test_results_unfair = df_u[df_u.subj.isin(test_subj_list)]
    test_metrics_f = compute_pareto_metrics(test_results_fair, loss_fct, attributes, all_only = True)
    test_metrics_u = compute_pareto_metrics(test_results_unfair, loss_fct, attributes, all_only = True)

    # Add the metrics
    for m in metrics_to_test:
        if test_metrics_f[m] < test_metrics_u[m]: success[m] = 1
        else: success[m] = 0
        
    return success


if __name__ == '__main__':
    #### Build the distribution of the data among the different subgroups
    # Initialization
    cols = attributes + ['pred_raw', 'label', 'pred']
    data_df = pd.DataFrame(columns = cols)
    range_atts = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    nb_patients_sg = [[1 , 16, 50 , 61 , 42, 23], 
                    [3 , 2 , 1  , 7  , 15, 14], 
                    [8 , 3 , 3  , 1  , 5 , 1], 
                    [37, 93, 136, 120, 67, 11], 
                    [2 , 12, 22 , 18 , 15, 1], 
                    [1 , 1 , 2  , 4  , 1 , 2]]

    # Loop on attributes
    for att_1 in range_atts[0]:
        for att_2 in range_atts[1]:
            nb = nb_patients_sg[att_1][att_2]
            sub_df = pd.DataFrame({'att_1' : [att_1] * nb,
                                'att_2' : [att_2] * nb,
                                'pred_raw' : [None] * nb,
                                'label' : list(np.random.binomial(1, 0.5, nb)),
                                'pred' : [None] * nb})
            data_df = pd.concat([data_df, sub_df])
    data_df = data_df.astype({'att_1' : 'int32',
                                'att_2' : 'int32',
                                'label' : 'int32'})
    
    # Initialization
    nb_success = {'MMPF' : 0,
                'MMPF_5' : 0, 'MMPF_10' : 0,
                'MMPF_size' : 0, 'MMPF_size_2' : 0,
                'MMPF_adapted' : 0, 'MMPF_adapted_2' : 0}
    n_scenario = 10000
    n_splits = 100
    n = 0

    # Get multiple seeds
    seeds_scenario = []
    for idx in range(n_scenario):
        seeds_scenario += [random.randint(0, 1e7)]

    # Loop on the number of scenarios
    for seed_sc in seeds_scenario:
        
        # Get scenario 
        df_f, df_u, _, _ = build_scenario(data_df, seed_sc)
        
        # Get multiple seeds
        seeds_splits = []
        for idx in range(n_splits):
            seeds_splits += [random.randint(0, 1e7)]
        
        # Loop on the splits
        for seed_sp in seeds_splits:
            print(n)
            n+=1
            
            # Check scenario 
            success = check_scenario(df_f, df_u, seed_sp)

            # Add success 
            nb_success['n'] = n
            for m in metrics_to_test:
                nb_success[m] += success[m]
                print(m, ' ', nb_success[m]/n)
                
        # Save dict
        with open('success.pkl', 'wb') as fp:
            pickle.dump(nb_success, fp)
            print('dictionary saved successfully to file')