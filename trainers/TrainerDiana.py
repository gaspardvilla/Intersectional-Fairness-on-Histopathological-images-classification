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
        self.weights = torch.ones(self.nb_subgroups) / self.nb_subgroups
        
        # Loop until reaching the maximum number of steps
        while K <= self.nb_STEPS:
            
            # Train the model and save the ckpts
            self._init_subtrainer(K)
            model.update_weights(self.weights)
            self.trainer.train_fit(model, TrainLoader, ValidationLoader)
            self.trainer.save_checkpoint(ckpt_path)
            
            # Update the weights and log the metrics
            current_error = self.trainer.model.get_avg_loss_subgroups()
            self._update_weights(current_error, K)
            self._log(current_error, K)
            
            # Loop update
            K += 1
            model = reset_model(model)
            
            
    def _update_weights(self, current_error : torch.Tensor,
                        K : int) -> None:
        # Update the weights at each iteration
        eta = 1 / torch.sqrt(torch.tensor(K))
        self.weights = self.weights * torch.exp(eta * current_error)
        self.weights = self.weights.detach()
    
    def test(self, model, TestLoader) -> None:
        self.trainer.test(model, TestLoader)
    
    
    def _init_subtrainer(self, K : int):
        if K == self.nb_STEPS:
            self.trainer = TrainerPL(max_epochs = self.max_epochs, 
                                     check_val_every_n_epoch = self.check_val_every_n_epochs, 
                                     logger = self.logger,
                                     sub_trainer = True)
        else:
            self.trainer = TrainerPL(max_epochs = 400, 
                                     check_val_every_n_epoch = self.check_val_every_n_epochs, 
                                     logger = self.logger,
                                     sub_trainer = True)
            
            
    def _log(self, current_error : torch.Tensor, 
             K : int) -> None:
        print(f'--------------- DIANA STEP : {K} ---------------')
        print(f'current error = {torch.max(current_error)}')
        if self.logger is not None:
            wandb.log({'K_step' : K,
                       'current_error' : torch.max(current_error)})