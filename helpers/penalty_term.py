import numpy as np
import torch


def compute_penalty_term(preds : torch.Tensor, 
                         prottected_attributes : torch.Tensor, 
                         pt_method : str, 
                         nb_classes : int, 
                         eps_1 : float) -> torch.Tensor:
    # Check which method should be used for the penalty term (positive / negative class)
    if pt_method == 'DF_pos': pt = compute_pt(preds, prottected_attributes, nb_classes, eps_1, 1)
    elif pt_method == 'DF_sum': pt = 0.5 * (compute_pt(preds, prottected_attributes, nb_classes, eps_1, 0) + compute_pt(preds, prottected_attributes, nb_classes, eps_1, 1))
    elif pt_method == 'DF_max': pt = torch.max(compute_pt(preds, prottected_attributes, nb_classes, eps_1, 0), compute_pt(preds, prottected_attributes, nb_classes, eps_1, 1))
    else: raise ValueError(f'pt_methpd argument should be \'DF_pos\', \'DF_sum\' or \'DF_max\'.')
    return pt


def compute_pt(preds : torch.Tensor, 
               prottected_attributes : torch.Tensor, 
               nb_classes : int, 
               eps_1 : float, 
               pc_index : int) -> torch.Tensor:
    # Initialization
    all_subgroups = torch.unique(prottected_attributes, dim = 0)
    nb_subgroups = len(all_subgroups)
    nb_preds = len(preds)
    
    # Counts on the class 1 
    counts_class = torch.zeros((nb_subgroups), dtype = torch.float)
    counts_total = torch.zeros((nb_subgroups), dtype = torch.float)
    
    # Loop on all the predictions
    for idx in range(nb_preds):
        
        # Get the index of current subgroup regarding the protected attributes
        idx_subgroup = np.where((all_subgroups == prottected_attributes[idx]).all(axis = 1))[0][0]
        
        # Update the counts for the concerned subgroup
        counts_total[idx_subgroup] = counts_total[idx_subgroup] + 1
        counts_class[idx_subgroup] = counts_class[idx_subgroup] + preds[idx][pc_index]
        
    
    # Compute the loss
    # Compute the main term for each subgroup - look formula
    alpha = 1 / nb_classes
    terms = (counts_class + alpha) / (counts_total + (alpha * nb_classes))
    
    # Compute the epsilon values for each subgroup
    epsilon_subgroups = torch.zeros(nb_subgroups, dtype = torch.float)
    for i in range(nb_subgroups):
        epsilon_row = torch.tensor(0.0)
        for j in range(nb_subgroups):
            if i != j:
                epsilon_row = torch.max(epsilon_row, torch.abs(torch.log(terms[i]) - torch.log(terms[j])))
                epsilon_row = torch.max(epsilon_row, torch.abs(torch.log(1 - terms[i]) - torch.log(1 - terms[j])))
        epsilon_subgroups[i] = epsilon_row
    epsilon = torch.max(epsilon_subgroups)
    
    # Return the penalty term
    penalty_term = torch.max(torch.tensor(0.0), (epsilon - eps_1))
    return penalty_term