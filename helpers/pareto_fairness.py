from torchmetrics.classification import BinaryF1Score
import lightning as L
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


def compute_MMPF_size(preds : torch.Tensor, 
                      targets : torch.Tensor, 
                      atts : torch.Tensor,
                      mmpf_args : int) -> dict:
    # Build the data frame so we can compute the metrics
    df = pd.DataFrame(data = {'pred_0' : preds[:, 0],
                              'pred_1' : preds[:, 1], 
                              'label_0' : targets[:, 0], 
                              'label_1' : targets[:, 1],
                              'age_' : atts[:,0], 
                              'race_' : atts[:,1],
                              'gender_' : atts[:,2]})
    
    # Evaluation
    avg_loss_df, nb_preds = _get_average_loss_subgroups(df, 'all_', ['age_', 'race_', 'gender_'])
    MMPF_metrics = _compute_pareto_metrics(avg_loss_df, 'all_', mmpf_args['N'], nb_preds, mmpf_args['N_subgroups'])
    
    # Return the wanted metric 
    return MMPF_metrics['MMPF_size_set_2']


def compute_pareto_metrics(pred : pd.DataFrame, 
                           protected_attributes : list,
                           all_only : bool = False) -> dict:
    # Check the good format of the pred data frame - To handle previous results
    preds = pred.copy()
    preds = preds.astype({'age_' : 'int32', 'race_' : 'int32', 'gender_' : 'int32', 'pred' : 'int32', 'label' : 'int32'})
    if 'pred_0' not in pred.columns.to_list():
        preds['pred_0'] = 1 - torch.Tensor(preds.pred.values)
        preds['pred_1'] = torch.Tensor(preds.pred.values)
        preds['label_0'] = 1 - torch.Tensor(preds.label.values)
        preds['label_1'] = torch.Tensor(preds.label.values)
    
    # Initialization
    pareto_fairness = {}
    N = len(preds)
    nb_subgroups_all = len(np.unique(preds[protected_attributes].values, axis = 0))

    # For the test of the results and the plots
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
        
        # # Init the loss function with the current weights
        # n0 = len(preds_subgroup[preds_subgroup.label_1 == 0])
        # n1 = len(preds_subgroup[preds_subgroup.label_1 == 1])
        # if (n0 != 0) & (n1 != 0): loss_fct = nn.CrossEntropyLoss(weight = torch.Tensor([(n0 + n1) / n0, (n0 + n1) / n1]))
        # else: loss_fct = nn.CrossEntropyLoss()
        loss_fct = nn.CrossEntropyLoss()
        
        # Compute the size and the average loss on the current subgroup
        avg_loss_df.loc[idx, 'size_'] = len(preds_subgroup)
        avg_loss_df.loc[idx, 'avg_loss'] = loss_fct(torch.Tensor(preds_subgroup[['pred_0', 'pred_1']].values), torch.Tensor(preds_subgroup[['label_0', 'label_1']].values)).item()
    
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
    nb_all_2 = max(min(int(N / (nb_subgroups_all)), avg_loss_df['size_'].max()), 1)
    nb_set_2 = max(min(int(nb_preds / (len(avg_loss_df))), avg_loss_df['size_'].max()), 1)
    metrics_dict[f'{set_}MMPF_size'] = avg_loss_df[avg_loss_df.size_ >= nb_all].iloc[0]['avg_loss']
    metrics_dict[f'{set_}MMPF_size_set'] = avg_loss_df[avg_loss_df.size_ >= nb_set].iloc[0]['avg_loss']
    metrics_dict[f'{set_}MMPF_size_2'] = avg_loss_df[avg_loss_df.size_ >= nb_all_2].iloc[0]['avg_loss']
    metrics_dict[f'{set_}MMPF_size_set_2'] = avg_loss_df[avg_loss_df.size_ >= nb_set_2].iloc[0]['avg_loss']

    # Extract the metric 'pareto_5' / 'pareto_10' / 'pareto_adapted_all' / 'pareto_adapted_set'
    # Initialization of some parameters
    nb_5 = max(int(0.05 * nb_preds), 1)
    nb_10 = max(int(0.1 * nb_preds), 1)

    current_nb_5 = 0
    current_nb_10 = 0
    current_nb_all = 0
    current_nb_set = 0
    current_nb_all_2 = 0
    current_nb_set_2 = 0

    pareto_5 = 0
    pareto_10 = 0
    pareto_all = 0
    pareto_set = 0
    pareto_all_2 = 0
    pareto_set_2 = 0

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
        
        # 'pareto_all'
        if current_nb_all_2 < nb_all_2:
            pareto_all_2 += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_all_2 += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # 'pareto_set'
        if current_nb_set_2 < nb_set_2:
            pareto_set_2 += avg_loss_df.iloc[idx]['avg_loss'] * avg_loss_df.iloc[idx]['size_']
            current_nb_set_2 += avg_loss_df.iloc[idx]['size_']
            stop_cond = 0
            
        # Update the idx
        idx += 1

    # Introduce the metric into the dictionnary
    metrics_dict[f'{set_}MMPF_5'] = pareto_5 / current_nb_5
    metrics_dict[f'{set_}MMPF_10'] = pareto_10 / current_nb_10
    metrics_dict[f'{set_}MMPF_adapted'] = pareto_all / current_nb_all
    metrics_dict[f'{set_}MMPF_adapted_set'] = pareto_set / current_nb_set
    metrics_dict[f'{set_}MMPF_adapted_2'] = pareto_all_2 / current_nb_all_2
    metrics_dict[f'{set_}MMPF_adapted_set_2'] = pareto_set_2 / current_nb_set_2
    
    # Return the dictionnary
    return metrics_dict