from pytorch_lightning.loggers import WandbLogger
from trainers.TrainerPL import TrainerPL
from helpers.helpers import reset_model
from models.Diana import Diana
import torch
import wandb


class TrainerDiana():
    
    def __init__(self, max_epochs : int,
                 check_val_every_n_epoch : int,
                 logger : WandbLogger,
                 nb_subgroups : int, 
                 nb_STEPS : int, **kwargs) -> None:
        # Initialization
        super().__init__()
        self.max_epochs = max_epochs
        self.check_val_every_n_epochs = check_val_every_n_epoch
        self.logger = logger
        self.nb_subgroups = nb_subgroups
        self.nb_STEPS = nb_STEPS
        
    
    def train_fit(self, model : Diana, 
                  TrainLoader, ValidationLoader, 
                  ckpt_path : str = None) -> None:
        # Initialization
        K = 1
        best_error = torch.Tensor([float('inf')])
        self.weights = torch.ones(self.nb_subgroups) / self.nb_subgroups
        
        # Loop until reaching the maximum number of steps
        while K <= self.nb_STEPS:
            
            # Train the model and save the ckpts
            self._init_subtrainer(K)
            model.update_weights(self.weights)
            self.trainer.train_fit(model, TrainLoader, ValidationLoader)
            
            # Check for improvement
            current_error = self.trainer.model.avg_loss_subgroups.clone()
            if torch.max(current_error) < torch.max(best_error):
            
                # Update the best error and save the ckpts
                best_error = current_error.clone()
                self.trainer.save_checkpoint(ckpt_path)
                
                # Save the weights for the best model
                self._save_weights(ckpt_path, self.trainer.model.subgroups)
            
            # Log the metrics and update weights
            self._update_weights(current_error, K)
            self._log(current_error, best_error, K)
            
            # Loop update
            K += 1
            model = reset_model(model)
            
            
    def _update_weights(self, current_error : torch.Tensor,
                        K : int) -> None:
        # Update the weights at each iteration
        eta = 1 / torch.sqrt(torch.tensor(K))
        self.weights = self.weights * torch.exp(eta * current_error)
        self.weights = self.weights.detach()
    
    
    def _init_subtrainer(self, K : int):
        if K == self.nb_STEPS:
            self.trainer = TrainerPL(max_epochs = self.max_epochs, 
                                     check_val_every_n_epoch = self.max_epochs, 
                                     logger = self.logger,
                                     sub_trainer = True)
        elif K == 1:
            self.trainer = TrainerPL(max_epochs = 100, 
                                     check_val_every_n_epoch = 100, 
                                     logger = self.logger,
                                     sub_trainer = True)
        else:
            self.trainer = TrainerPL(max_epochs = 100, 
                                     check_val_every_n_epoch = 100, 
                                     logger = None,
                                     sub_trainer = True)
            
            
    def _log(self, current_error : torch.Tensor,
             best_error : torch.Tensor,
             K : int) -> None:
        print(f'--------------- DIANA STEP : {K} ---------------')
        print(f'current error = {torch.max(current_error)}')
        print(f'best error = {torch.max(best_error)}')
        if self.logger is not None:
            wandb.log({'K_step' : K,
                       'current_error' : torch.max(current_error),
                       'best_error' : torch.max(best_error)})
            
            
    def _save_weights(self, ckpt_path : str, subgroups : torch.Tensor) -> None:
        split_path = ckpt_path.split('/')
        
        # Save weights
        split_path[-1] = 'best_weights.pt'
        torch.save(self.weights, '/'.join(split_path))
    
        # Save corresponding subgroups
        split_path[-1] = 'subgroups.pt'
        torch.save(subgroups, '/'.join(split_path))
        
    
    def test(self, model, TestLoader) -> None:
        self.trainer.test(model, TestLoader)