from models.Trainers import TrainerPL, TrainerMartinez, TrainerDiana
from pytorch_lightning.loggers import WandbLogger
from dataprocess.dataclass import Data
from models.Baseline import Baseline
from models.Martinez import Martinez
from models.Foulds import Foulds
from models.Diana import Diana


def init(**kwargs):
    
    # Load the data
    data = Data(**kwargs)
    print(data)
    
    # Initialization of the model
    if kwargs['model'] == 'Baseline': model = Baseline(data.nb_features, **kwargs)
    elif kwargs['model'] == 'Martinez': model = Martinez(data.nb_features, **kwargs)
    elif kwargs['model'] == 'Foulds': model = Foulds(data.nb_features, **kwargs)
    elif kwargs['model'] == 'Diana': model = Diana(data.nb_features, **kwargs)
        
    # Reduce the configuration arguments for WandB
    config = {'job_id': kwargs['job_id'],
              'model': kwargs['model'],
              'task': kwargs['task'],
              'cancer': kwargs['cancer'],
              'add_protected_attributes': kwargs['add_protected_attributes'],
              'pt_method': kwargs['pt_method'],
              'eps_1': kwargs['eps_1'],
              'lambda_': kwargs['lambda_'],
              'alpha_': kwargs['alpha_'],
              'ID' : kwargs['ID']}
    
    # Wandb 
    if kwargs['allow_wandb']: 
        wandb = WandbLogger(name = f'{kwargs["model"]}_{kwargs["task"]}_{kwargs["cancer"]}', 
                            log_model = 'all', 
                            project = 'Fairness',
                            save_dir = 'logs/wandb',
                            config = config)
    else: wandb = None
    
    # Get the adapted Trainer
    if kwargs['model'] == 'Martinez': trainer = TrainerMartinez(max_epochs = kwargs['nb_epochs'],
                                                                check_val_every_n_epoch = kwargs['check_val'],
                                                                logger = wandb,
                                                                nb_subgroups = data.nb_subgroups,
                                                                nb_STEPS = kwargs['NB_STEPS'],
                                                                alpha = kwargs['alpha_'])
    elif kwargs['model'] == 'Diana': trainer = TrainerDiana(max_epochs = kwargs['nb_epochs'],
                                                            check_val_every_n_epoch = kwargs['check_val'],
                                                            logger = wandb,
                                                            train_protected_atts = list(enumerate(data.TrainLoader))[0][1][2],
                                                            nb_STEPS = kwargs['NB_STEPS'])
    else: trainer = TrainerPL(max_epochs = kwargs['nb_epochs'],
                              check_val_every_n_epoch = kwargs['check_val'],
                              logger = wandb)
        
    # Return
    return data, model, wandb, trainer