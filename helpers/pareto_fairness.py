import lightning as L
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


def compute_pareto_metrics(pred : pd.DataFrame, 
                           loss_fct,
                           protected_attributes : list,
                           all_only : bool = False) -> dict:
    # Initialization
    preds = pred.copy()
    pareto_fairness = {}
    N = len(preds)
    nb_subgroups_all = len(np.unique(preds[protected_attributes].values, axis = 0))

    # Compute the loss for all the predictions
    preds['label_raw'] = list(torch.cat([1 - torch.Tensor(preds.label.values).unsqueeze(0), torch.Tensor(preds.label.values).unsqueeze(0)], dim = 0).T)
    preds['loss'] = preds.apply(lambda row: loss_fct(torch.from_numpy(np.reshape(list(row['pred_raw']), (1, 2))), torch.from_numpy(np.reshape(list(row['label_raw']), (1, 2)))).item(), axis = 1)

    if all_only : sets_to_evaluate = ['all_']
    else: sets_to_evaluate = ['all_', 'train_', 'val_', 'test_', 'full_test_']

    # Loop on all possible set and get the paretor fairness scores for each
    for set_ in sets_to_evaluate:
        
        # Get the avg_loss_df for the current set 
        avg_loss_df, nb_preds = _get_average_loss_subgroups(preds, set_, protected_attributes)
        
        # Compute the metrics and add them to the dictionnary
        pareto_fairness.update(_compute_pareto_metrics(avg_loss_df, set_, N, nb_preds, nb_subgroups_all))
    
    # Return the dictionnary with all the pareto fairness measures
    return pareto_fairness


def _get_average_loss_subgroups(preds : pd.DataFrame, 
                                set_ : str,
                                protected_attributes : list):
    # Filter the preds with the desired data set
    sub_preds = _select_preds(preds, set_)
    
    # Initialization of the dataframe
    avg_loss_df = pd.DataFrame(data = np.unique(sub_preds[protected_attributes].values, axis = 0),
                               columns = protected_attributes)
    avg_loss_df['avg_loss'] = -1.0
    avg_loss_df['size_'] = -1.0
    
    # For each subgroup, compute its size and the average loss on it
    for idx in range(len(avg_loss_df)):
        
        # Get all the predictions from the subgroup
        cond = sub_preds[protected_attributes[0]] == avg_loss_df.iloc[idx][protected_attributes[0]]
        for i in range(len(protected_attributes) - 1): 
            cond = cond & (sub_preds[protected_attributes[i+1]] == avg_loss_df.iloc[idx][protected_attributes[i+1]])
        preds_subgroup = sub_preds[cond]
        
        # Compute the size and the average loss on the current subgroup
        avg_loss_df.loc[idx, 'size_'] = len(preds_subgroup)
        avg_loss_df.loc[idx, 'avg_loss'] = np.sum(preds_subgroup.loss.values).item() / len(preds_subgroup)
    
    # Return the dataframe
    return avg_loss_df, len(sub_preds)
    
    
def _select_preds(preds : pd.DataFrame,
                  set_ : str) -> pd.DataFrame:
    # Filter the preds with the desired data set
    if set_ == 'all_': sub_preds = preds.copy()
    elif set_ == 'train_': sub_preds = preds[preds.set == 'train']
    elif set_ == 'val_': sub_preds = preds[preds.set == 'val']
    elif set_ == 'test_': sub_preds = preds[preds.set == 'test']
    elif set_ == 'full_test_': sub_preds = preds[preds.set.isin(['val', 'test'])]
    return sub_preds


def _compute_pareto_metrics(avg_loss_df : pd.DataFrame, 
                            set_ : str, 
                            N : int,
                            nb_preds : int,
                            nb_subgroups_all : int) -> dict:
    # Initialization
    if set_ == 'all_': set_ = ''
    metrics_dict = {}
    
    # Sort avg_loss_df by loss (descending) and size (descending)
    avg_loss_df.sort_values(['avg_loss', 'size_'], ascending = [False, False], inplace = True, ignore_index = True)
    
    # Extract the metric 'pareto_classic' - Maximum Average Loss (taken on the largest subgroups if equality)
    metrics_dict[f'{set_}MMPF'] = avg_loss_df.iloc[0]['avg_loss']
    
    # Extract the metric 'pareto_minimum_size' - Maximum average loss on the subgroups of size at least larger than (1/nb_subgroups) * nb_preds
    nb_all = max(min(int(N / (2 * nb_subgroups_all)), avg_loss_df['size_'].max()), 1)
    nb_set = max(min(int(nb_preds / (2 * len(avg_loss_df))), avg_loss_df['size_'].max()), 1)
    metrics_dict[f'{set_}MMPF_size'] = avg_loss_df[avg_loss_df.size_ >= nb_all].iloc[0]['avg_loss']
    metrics_dict[f'{set_}MMPF_size_set'] = avg_loss_df[avg_loss_df.size_ >= nb_set].iloc[0]['avg_loss']

    # Extract the metric 'pareto_5' / 'pareto_10' / 'pareto_adapted_all' / 'pareto_adapted_set'
    # Initialization of some parameters
    nb_5 = max(int(0.05 * nb_preds), 1)
    nb_10 = max(int(0.1 * nb_preds), 1)

    current_nb_5 = 0
    current_nb_10 = 0
    current_nb_all = 0
    current_nb_set = 0

    pareto_5 = 0
    pareto_10 = 0
    pareto_all = 0
    pareto_set = 0

    # Loop on the avg_loss_df (sorted)
    stop_cond = 0
    idx = 0
    while stop_cond == 0:
        stop_cond = 1
        
        # 'pareto_5'
        if current_nb_5 < nb_5:
            pareto_5 += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_5 += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # 'pareto_10'
        if current_nb_10 < nb_10:
            pareto_10 += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_10 += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # 'pareto_all'
        if current_nb_all < nb_all:
            pareto_all += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_all += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # 'pareto_set'
        if current_nb_set < nb_set:
            pareto_set += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_set += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # Update the idx
        idx += 1

    # Introduce the metric into the dictionnary
    metrics_dict[f'{set_}MMPF_5'] = pareto_5 / current_nb_5
    metrics_dict[f'{set_}MMPF_10'] = pareto_10 / current_nb_10
    metrics_dict[f'{set_}MMPF_adapted'] = pareto_all / current_nb_all
    metrics_dict[f'{set_}MMPF_adapted_set'] = pareto_set / current_nb_set
    
    # Return the dictionnary
    return metrics_dict


# def _compute_pareto_set(preds : pd.DataFrame, 
#                         set_ : str, 
#                         protected_attributes : list):
#     # Filter the preds with the desired data set
#     sub_preds = _select_preds(preds, set_)
    
#     # Initialization
#     n_5 = max(int(0.05 * len(sub_preds)), 1)
#     n_10 = max(int(0.1 * len(sub_preds)), 1)
#     pareto_5 = 0
#     pareto_10 = 0
#     current_n_5 = 0
#     current_n_10 = 0
#     discriminated_subgroups = {f'{set_}5%': [], f'{set_}10%' : []}
    
#     # Create the dataframe that will contains all pareto fairness measures
#     pareto_df = pd.DataFrame(data = torch.unique(torch.Tensor(sub_preds[protected_attributes].values), dim = 0),
#                                 columns = protected_attributes)
#     pareto_df['size'] = None
#     pareto_df['fairness_score'] = None

#     # Loop on all the different subgroups in the data set
#     for idx in range(len(pareto_df)):
        
#         # Restrcict to the preds in this subgroup
#         indices = sub_preds[protected_attributes[0]] == pareto_df.iloc[idx][protected_attributes[0]]
#         for i in range(len(protected_attributes) - 1): 
#             indices = indices & (sub_preds[protected_attributes[i+1]] == pareto_df.iloc[idx][protected_attributes[i+1]])
#         sub_df = sub_preds[indices]
        
#         # Get the size of the subgroup and get the average loss in it
#         pareto_df.loc[idx, 'size'] = len(sub_df)
#         pareto_df.loc[idx, 'fairness_score'] = np.sum(sub_df.loss.values).item() / len(sub_df)
    
#     # Compute the pareto fairness with rate raw / 5% / 10
#     pareto_df.sort_values(['fairness_score', 'size'], ascending = [False, False], inplace = True, ignore_index = True)
#     pareto_df['raw_score'] = pareto_df['size'] * pareto_df.fairness_score
#     pareto_raw = pareto_df.iloc[0]['fairness_score']
#     idx = 0
#     while current_n_10 < n_10:
#         if current_n_5 < n_5:
#             pareto_5 += pareto_df.iloc[idx]['raw_score']
#             current_n_5 += pareto_df.iloc[idx]['size']
#             discriminated_subgroups[f'{set_}5%'] += [list(pareto_df.iloc[idx][protected_attributes].values)]
#         pareto_10 += pareto_df.iloc[idx]['raw_score']
#         current_n_10 += pareto_df.iloc[idx]['size']
#         discriminated_subgroups[f'{set_}10%'] += [list(pareto_df.iloc[idx][protected_attributes].values)]
#         idx += 1
#     pareto_5 = pareto_5 /current_n_5
#     pareto_10 = pareto_10 / current_n_10
    
#     # Return all the pareto fairness
#     return pareto_raw, pareto_5, pareto_10