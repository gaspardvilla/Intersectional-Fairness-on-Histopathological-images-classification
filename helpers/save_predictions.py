from helpers.pareto_fairness import compute_pareto_metrics
from helpers.penalty_term import compute_penalty_term
from pytorch_lightning.loggers import WandbLogger
from dataprocess.dataclass import Data
import lightning as L
import numpy as np
import argparse
import torch
import os


def save_predictions(args : argparse.Namespace, 
                     data : Data, 
                     model : L.LightningModule,
                     wandb : WandbLogger =  None) -> None:
    
    # Get the trained model
    ckpts = torch.load(args.ckpt_path)
    model.load_state_dict(ckpts['state_dict'])
    model.eval()

    # Evaluation on the whole data set
    preds_df = data.references
    preds_df['pred_raw'] = None
    preds_df['pred'] = None
    preds_df['set'] = None
    for idx in range(len(preds_df)):
        
        # Get the data and its prediction
        X_idx = data.data[idx].unsqueeze(0)
        with torch.no_grad(): 
            y_pred = model(X_idx)
        
        # Update the dataframe
        preds_df.loc[idx, 'pred_raw'] = [y_pred]
        preds_df.loc[idx, 'pred'] = torch.argmax(y_pred, dim = 1)[0].item()
        if preds_df.loc[idx, 'subj'] in list(data.TrainSet.references.subj): preds_df.loc[idx, 'set'] = 'train'
        elif preds_df.loc[idx, 'subj'] in list(data.ValidationSet.references.subj): preds_df.loc[idx, 'set'] = 'val'
        elif preds_df.loc[idx, 'subj'] in list(data.TestSet.references.subj): preds_df.loc[idx, 'set'] = 'test'
        
    # Save the dataframe with the predictions
    preds_path = f'results/preds/{args.run_path}'
    if not os.path.exists(preds_path): os.makedirs(preds_path)
    preds_df.to_pickle(f'{preds_path}/results.pkl')
    
    # Extract the Pareto fairness metrics
    pareto_fairness_dict = compute_pareto_metrics(preds_df, model.loss_fct, protected_attributes = args.protected_attributes)
    if wandb is not None: wandb.log_metrics(pareto_fairness_dict)
    
    # # Extract the penalty term value
    # wandb.log_metrics({'pt' : compute_penalty_term(torch.from_numpy(np.reshape(list(preds_df.pred_raw), (len(preds_df), 2))), 
    #                                                 torch.tensor(preds_df[args.protected_attributes].values), 
    #                                                 args.pt_method, args.nb_classes, args.eps_1),
    #                     'train/pt' : compute_penalty_term(torch.from_numpy(np.reshape(list(preds_df[preds_df.set == 'train'].pred_raw), (len(preds_df[preds_df.set == 'train']), 2))), 
    #                                                         torch.tensor(preds_df[preds_df.set == 'train'][args.protected_attributes].values), 
    #                                                         args.pt_method, args.nb_classes, args.eps_1),
    #                     'validation/pt' : compute_penalty_term(torch.from_numpy(np.reshape(list(preds_df[preds_df.set == 'val'].pred_raw), (len(preds_df[preds_df.set == 'val']), 2))), 
    #                                                     torch.tensor(preds_df[preds_df.set == 'val'][args.protected_attributes].values), 
    #                                                     args.pt_method, args.nb_classes, args.eps_1),
    #                     'test/pt' : compute_penalty_term(torch.from_numpy(np.reshape(list(preds_df[preds_df.set == 'test'].pred_raw), (len(preds_df[preds_df.set == 'test']), 2))), 
    #                                                         torch.tensor(preds_df[preds_df.set == 'test'][args.protected_attributes].values), 
    #                                                         args.pt_method, args.nb_classes, args.eps_1)})