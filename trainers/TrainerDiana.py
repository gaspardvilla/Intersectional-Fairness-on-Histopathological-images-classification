from pytorch_lightning.loggers import WandbLogger
from trainers.TrainerPL import TrainerPL


class TrainerDiana():
    
    def __init__(self, max_epochs : int,
                 check_val_every_n_epoch : int,
                 logger : WandbLogger, 
                 sub_trainer : bool = False, **kwargs) -> None:
        # Initialization
        super().__init__()
        self.trainer = L.Trainer(max_epochs = max_epochs,
                                 check_val_every_n_epoch = check_val_every_n_epoch,
                                 devices = 1, 
                                 accelerator = 'auto',
                                 log_every_n_steps = 1,
                                 num_sanity_val_steps = -1,
                                 logger = logger,
                                 default_root_dir = 'logs/lightning')
        self.sub_trainer = sub_trainer