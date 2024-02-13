from pytorch_lightning.loggers import WandbLogger
from trainers.TrainerPL import TrainerPL
from helpers.helpers import reset_model
from models.Martinez import Martinez
import torch
import wandb


class TrainerMartinez():
    
    def __init__(self, max_epochs : int,
                 check_val_every_n_epoch : int,
                 logger : WandbLogger,
                 nb_subgroups : int, 
                 nb_STEPS : int,
                 alpha : float, **kwargs) -> None:
        # Initialization
        super().__init__()
        self.max_epochs = max_epochs
        self.check_val_every_n_epochs = check_val_every_n_epoch
        self.logger = logger
        self.nb_subgroups = nb_subgroups
        self.nb_STEPS = nb_STEPS
        self.alpha = alpha
        
    
    def train_fit(self, model : Martinez, 
                  TrainLoader, ValidationLoader, 
                  ckpt_path : str = None) -> None:
        # Initialization
        K = 1
        best_risk = torch.Tensor([float('inf')])
        self.weights = torch.ones(self.nb_subgroups) / self.nb_subgroups
        
        # Loop until reaching the maximum number of steps
        while K <= self.nb_STEPS:
            
            # Train the model
            self._init_subtrainer(K)
            model.update_weights(self.weights)
            model.update_ckpt_path(ckpt_path)
            self.trainer.train_fit(model, TrainLoader, ValidationLoader)
            
            # Check for improvement
            current_risk = model.best_risk
            self._update_weights(current_risk, best_risk, K)
            if torch.max(current_risk) < torch.max(best_risk):
                
                # Update the best risk and save the chechpoints
                best_risk = current_risk.clone()
                self.trainer.model.load_best_train()
                self.trainer.save_checkpoint(ckpt_path)
                
            # Log the metrics
            self._log(current_risk, best_risk, K)
            
            # Loop update
            K += 1
            model = reset_model(model)
            
            
    def _update_weights(self, current_risk : torch.Tensor,
                        best_risk : torch.Tensor,
                        K : int) -> None:
        # Get index where risk is worse than the best one
        ones = torch.zeros(self.nb_subgroups)
        ones[current_risk >= best_risk] = 1
        
        # Update the weights 
        self.weights = (self.alpha * self.weights) + (((1 - self.alpha) / K) * ones)
        self.weights = self.weights / self.weights.sum()
        
    
    def test(self, model, TestLoader) -> None:
        self.trainer.test(model, TestLoader)
    
    
    def _init_subtrainer(self, K : int):
        if K == self.nb_STEPS:
            self.trainer = TrainerPL(max_epochs = self.max_epochs, 
                                     check_val_every_n_epoch = self.check_val_every_n_epochs, 
                                     logger = self.logger,
                                     sub_trainer = True)
        else:
            self.trainer = TrainerPL(max_epochs = 100, 
                                     check_val_every_n_epoch = self.check_val_every_n_epochs, 
                                     logger = self.logger,
                                     sub_trainer = True)
            
            
    def _log(self, current_risk : torch.Tensor, 
             best_risk : torch.Tensor, 
             K : int) -> None:    
        print(f'--------------- MARTINEZ STEP : {K} ---------------')
        print(f'alpha = {self.alpha}')
        print(f'current risk = {torch.max(current_risk)}')
        print(f'best risk = {torch.max(best_risk)}')
        print(f'risk loss = {current_risk.sum()}')
        if self.logger is not None:
            wandb.log({'K_step' : K,
                       'current_risk' : torch.max(current_risk),
                       'best_risk' : torch.max(best_risk),
                       'risk_loss' : current_risk.sum()})