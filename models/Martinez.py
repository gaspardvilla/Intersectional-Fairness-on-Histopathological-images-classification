from models.Baseline import Baseline
from torch.autograd import Variable
import torch.nn as nn
import torch
import os


class Martinez(Baseline):
    
    def __init__(self, in_features : int, **kwargs):
        # Initialization
        super().__init__(in_features, **kwargs)
        self.alpha = kwargs['alpha_']
        self.risk_tensor = None
        self.loss_fct_elementwise = nn.CrossEntropyLoss(weight = kwargs['train_weights'], reduction = 'none')
        self.lr = 1e-2
        self.decay = 0.25
        self.best_risk = torch.Tensor([float('inf')])
        self.ckpt_path = None
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode = 'min',
                                                               factor = self.decay)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'train/risk'}}
        
    
    def training_step(self, train_batch, batch_idx):
        # Extract the data from the batch and evaluation
        X, y, atts = train_batch
        X_ = Variable(X, requires_grad = True)
        y_hat = self.forward(X_)
        
        # Compute the metrics and return the loss
        metrics = self._compute_metrics(y_hat, y, atts, 'train')
        metrics['train/risk'] = torch.max(self.risk_tensor)
        self.log_dict(metrics, on_step = False, on_epoch = True)
        
        # Check if there an improvement and return the loss
        self._check_for_improvement()
        return metrics['train/loss']
        
        
    def _compute_loss(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor, task : str):
        # Compute the loss during training or not
        if task == 'train': loss = self._compute_loss_risk(preds, targets, atts)
        else: loss = self.loss_fct(preds, targets)
        
        # Return the loss
        return loss
    
    
    def _compute_loss_risk(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor):
        # Initialization
        self.subgroups = torch.unique(atts, dim = 0)
        indicator = torch.zeros(len(preds))
        risk_tensor = torch.zeros(len(self.subgroups))

        # Loop on the subgroups
        for idx in range(len(self.subgroups)):
            
            # Get the indices of each element of the current subgroup
            cond_sg = torch.all(atts == self.subgroups[idx], dim = 1)
            
            # Change the indicator value for the loss
            indicator[cond_sg] = self.weights[idx] / len(targets[cond_sg])
            
            # Update the risk vector
            risk_tensor[idx] = self.loss_fct(preds[cond_sg], targets[cond_sg])
            
        # Compute the weighted mean loss of the predictions and return the loss
        self.risk_tensor = risk_tensor.clone()# * self.weights
        loss = torch.sum(self.loss_fct_elementwise(preds, targets) * indicator)
        return loss
        
        
    def update_ckpt_path(self, ckpt_path : str) -> None:
        if self.ckpt_path is None:
            split_path = ckpt_path.split('/')
            split_path.insert(len(split_path) - 1, 'prov')
            self.ckpt_path = '/'.join(split_path)
            
    
    def _check_for_improvement(self) -> None:
        if torch.max(self.risk_tensor) < torch.max(self.best_risk): 
            self.best_risk = self.risk_tensor.clone()
            self._save()
    
    
    def load_best_train(self):
        if os.path.isfile(self.ckpt_path):
            ckpts = torch.load(self.ckpt_path)
            self.load_state_dict(ckpts['state_dict'])
        else: print('No pre-trained model - sanity check step for validation?')
        
        
    def update_weights(self, weights : torch.Tensor) -> None:
        self.weights = weights.clone()
            
            
    def _save(self):
        self.trainer.save_checkpoint(self.ckpt_path)
    
        
    def name(self) -> str:
        return 'Martinez'