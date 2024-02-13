from pytorch_lightning.loggers import WandbLogger
import lightning as L


class TrainerPL(L.Trainer):
    
    def __init__(self, max_epochs : int,
                 check_val_every_n_epoch : int,
                 logger : WandbLogger, 
                 sub_trainer : bool = False, **kwargs) -> None:
        # Initialization
        super().__init__(max_epochs = max_epochs,
                         check_val_every_n_epoch = check_val_every_n_epoch,
                         devices = 1, 
                         accelerator = 'auto',
                         log_every_n_steps = 1,
                         num_sanity_val_steps = -1,
                         logger = logger,
                         default_root_dir = 'logs/lightning')
        self.sub_trainer = sub_trainer
        
        
    def train_fit(self, model,
                  TrainLoader, ValidationLoader,
                  ckpt_path : str = None) -> None:
        # Fit the model using Pytorch Lightning trainer 
        self.fit(model, TrainLoader, ValidationLoader)
        
        # In case we are not a sub trainer we can svae the checkpoints
        if not self.sub_trainer:
            self.save_checkpoint(ckpt_path)