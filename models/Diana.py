from models.Baseline import Baseline
import torch.nn as nn
import torch
import os


class Diana(Baseline):
    
    def __init__(self, in_features : int, **kwargs):
        # Initialization
        super().__init__(in_features, **kwargs)
        self.alpha = kwargs['alpha_']
        self.avg_loss_subgroups = None
        self.loss_fct_elementwise = nn.CrossEntropyLoss(reduction = 'none')
      
        
    def _compute_loss(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor, task : str):
        # Compute the loss during training or not
        if task == 'train': loss = self._compute_avg_loss_subgroups(preds, targets, atts)
        else: loss = self.loss_fct(preds, targets)
        
        # Return the loss
        return loss
    
    
    def _compute_avg_loss_subgroups(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor):
        # Initialization
        subgroups = torch.unique(atts, dim = 0)
        indicator = torch.zeros(len(preds))
        avg_loss_subgroups = torch.zeros(len(subgroups))

        # Loop on the subgroups
        for idx in range(len(subgroups)):
            
            # Get the indices of each element of the current subgroup
            cond_sg = torch.all(atts == subgroups[idx], dim = 1)
            
            # Change the indicator value for the loss
            indicator[cond_sg] = self.weights[idx] / len(targets[cond_sg])
            
            # Update the avg loss subgroups tensor
            avg_loss_subgroups[idx] = self.loss_fct(preds[cond_sg], targets[cond_sg])
            
        # Compute the weighted mean loss of the predictions
        self.avg_loss_subgroups = avg_loss_subgroups.clone()# * self.weights
        loss = torch.sum(self.loss_fct_elementwise(preds, targets) * indicator)
        
        # Return the loss
        return loss
    
    
    def get_avg_loss_subgroups(self):
        return self.avg_loss_subgroups.clone()
    
    
    # def load_pretrained(self, ckpt_path : str):
    #     # Check the checkpoints exist
    #     if os.path.exists(ckpt_path):        
    #         ckpts = torch.load(ckpt_path)
    #         self.load_state_dict(ckpts['state_dict'])
    #     else: print('--------- Random initialization used [NO CHECKPOINTS] ---------')
     
        
    def update_weights(self, weights : torch.Tensor) -> None:
        self.weights = weights.clone()
        
        
    def name(self) -> str:
        return 'Diana'