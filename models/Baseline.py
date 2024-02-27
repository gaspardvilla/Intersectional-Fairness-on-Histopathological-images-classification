from helpers.pareto_fairness import compute_MMPF_size
from models.Layers import MLP, MILAttention
from torch.autograd import Variable
import torchmetrics as tm
import lightning as L
import torch.nn as nn
import torch


class Baseline(L.LightningModule):
    
    def __init__(self, in_features : int, **kwargs):
        # Initialization
        super().__init__()
        self.nb_classes = kwargs['nb_classes']
        self.dropout = kwargs['dropout']
        self.loss_fct = nn.CrossEntropyLoss()
        self.mmpf_args = kwargs['MMPF_args']
        
        # Tracking lists
        # Validation
        self.targets_val = []
        self.preds_val = []
        self.prot_atts_val = []
        # Test
        self.targets_test = []
        self.preds_test = []
        self.prot_atts_test = []
        
        # Initialization of the layers of the model
        self.attention_layer = MILAttention(featureLength = in_features, 
                                            featureInside = kwargs['MIL_latent_size'])
        self.classifier = MLP(in_features = in_features, 
                              fc_latent_size = kwargs['fc_latent_size'], 
                              nb_classes = self.nb_classes, 
                              dropout = self.dropout)
        
        # For a reset of the model
        self.in_features = in_features
        self.kwargs = kwargs


    def forward(self, input):
        weights = self.attention_layer(input)
        features_MIL = torch.bmm(weights, input).squeeze(1)
        output = self.classifier(features_MIL)
        return output
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
    

    def training_step(self, train_batch, batch_idx):
        # Extract the data from the batch and evaluation
        X, y, atts = train_batch
        X_ = Variable(X, requires_grad = True)
        y_hat = self.forward(X_)
        
        # Compute the metrics and return the loss
        metrics = self._compute_metrics(y_hat, y, atts, 'train')
        if ((self.trainer.current_epoch + 1) % 50 == 0) or self.trainer.is_last_batch:
            self.log_dict(metrics, on_step = False, on_epoch = True)
        return metrics['train/loss']
    
    
    def validation_step(self, validation_batch, batch_idx):
        # Extract the data from the batch and evaluation
        X, y, atts = validation_batch
        y_hat = self.forward(X)
        
        # Keep track of the prediction made at each batch
        self.targets_val.append(y)
        self.preds_val.append(y_hat)
        self.prot_atts_val.append(atts)
        
        
    def test_step(self, test_batch, batch_idx):
        # Extract the data from the batch and evaluation
        X, y, atts = test_batch
        y_hat = self.forward(X)
        
        # Keep track of the prediction made at each batch
        self.targets_test.append(y)
        self.preds_test.append(y_hat)
        self.prot_atts_test.append(atts)
        
        
    def on_validation_epoch_end(self):
        # Get all the preds from the validation and compute all metrics
        targets_val = torch.stack(self.targets_val).squeeze()
        preds_val = torch.stack(self.preds_val).squeeze()
        prot_atts_val = torch.stack(self.prot_atts_val).squeeze()
        
        # Compute the metrics
        metrics = self._compute_metrics(preds_val, targets_val, prot_atts_val, 'validation')
        self.log_dict(metrics, on_step = False, on_epoch = True)
        
        # Free the memory
        self.targets_val.clear()
        self.preds_val.clear()
        self.prot_atts_val.clear()
        
        
    def on_test_epoch_end(self):
        # Get all the preds from the validation and compute all metrics
        targets_test = torch.stack(self.targets_test).squeeze()
        preds_test = torch.stack(self.preds_test).squeeze()
        prot_atts_test = torch.stack(self.prot_atts_test).squeeze()
        
        # Compute the metrics
        metrics = self._compute_metrics(preds_test, targets_test, prot_atts_test, 'test')
        self.log_dict(metrics, on_step = False, on_epoch = True)
        
        # Free the memory
        self.targets_test.clear()
        self.preds_test.clear()
        self.prot_atts_test.clear()
        
    
    def _compute_metrics(self, preds : torch.Tensor, targets : torch.Tensor, 
                         atts : torch.Tensor, task : str):
        # Initialization
        preds_1D = torch.argmax(preds, dim = 1)
        targets_1D = torch.argmax(targets, dim = 1)
                
        # Compute the metrics and return them
        if task == 'train':
            metrics = {f'{task}/Accuracy' : tm.Accuracy(task = 'multiclass', num_classes = self.nb_classes)(preds_1D, targets_1D),
                       f'{task}/loss' : self._compute_loss(preds, targets, atts, task)}
        else:
            metrics = {f'{task}/Accuracy' : tm.Accuracy(task = 'multiclass', num_classes = self.nb_classes)(preds_1D, targets_1D),
                       f'{task}/MMPF_size' : compute_MMPF_size(preds, targets_1D, atts, self.mmpf_args, self.loss_fct),
                       f'{task}/loss' : self._compute_loss(preds, targets, atts, task)}
        return metrics
    
    
    def _compute_loss(self, preds : torch.Tensor, targets : torch.Tensor, atts : torch.Tensor, task : str):
        # Compute the loss
        loss = self.loss_fct(preds, targets)
        return loss
    
    
    def name(self) -> str:
        return 'Baseline'