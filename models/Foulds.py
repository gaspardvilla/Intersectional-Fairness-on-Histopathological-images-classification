from helpers.penalty_term import compute_penalty_term
from models.Baseline import Baseline
import lightning as L
import torch


class Foulds(Baseline):
    
    def __init__(self, in_features : int, **kwargs):
        # Initialization
        super().__init__(in_features, **kwargs)
        self.lambda_ = kwargs['lambda_']
        self.pt_method = kwargs['pt_method']
        self.eps_1 = kwargs['eps_1']
        
        self.task = kwargs['task']
        self.cancer = kwargs['cancer']
        self.atts = kwargs['add_protected_attributes']
        
        
    def _compute_loss(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor, task : str):
        # Compute the loss with the fairness penalty term
        penalty_term = compute_penalty_term(preds, atts, self.pt_method, self.nb_classes, self.eps_1)
        loss = self.loss_fct(preds, targets) + (self.lambda_ * penalty_term)
        return loss
    
    
    def name(self) -> str:
        return 'Foulds'