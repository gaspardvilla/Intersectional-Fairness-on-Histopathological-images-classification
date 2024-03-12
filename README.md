# Intersectional-Fairness-on-Histopathological-images-classification

***
## Introduction

This repository contains all the code used to make all the experiments presented in the context of the Master Thesis. The corresponding report is attached to this repository in the PDF file: `Gaspard Villa - Master Thesis.pdf`. 

***
## Code architeture

• `pipeline.py` : Code used to train and test a specific model on a specific task given the inputs given in the scripts files. \
• `scripts/` : Folder containing all the scripts used to train all the methods on all the different classification tasks studied in the scope of the report.\

#### Method classes
• `models/Layers.py` : Contains the class `MLP` and `MILAttention` that represents the common architecture for the four different training methods. \
• `models/Baseline.py` : Contains the class of the `Baseline` training method against which the three other will be evaluated.\
• `models/Foulds.py` : Contains the class `Foulds` corresponding to the EDF training method presented in the report.\
• `models/Diana.py` : Contains the class `Diana` corresponding to the MinimaxFair training method presented in the report.\
• `models/Martinez.py` : Contains the class of the `Martinez` corresponding to the APStar training method presented in the report.\

#### Trainers classes
• `trainers/TrainerPL.py` : Contains the class `TrainerPL` which corresponds to the trainer of PyTorch Lightning adapted to our situation where it can be called many times by the other trainers.\
• `trainers/TrainerDiana.py` : Contains the class `TrainerDiana` that corresponds to a handcrafted trainer for the MinimaxFair training method (with the re-weighting method at each iteration), it also handle PyTorch Lightning trainer calling at each iteration the class `TrainerPL`.\
• `trainers/TrainerMartinez.py` : Contains the class `TrainerMartinez` that corresponds to a handcrafted trainer for the APStar training method (with the re-weighting method at each iteration), it also handle PyTorch Lightning trainer calling at each iteration the class `TrainerPL`.\

#### Data processing
• `dataprocess/dataloader.py` : Contains the function `load_data` that load the valid data in a suitable format for the actual code by removing all the sample without all the three protected attributes information.\
• `dataprocess/dataclass.py` : Contains the classes `DataClass` and `Data` corresponding respectively to the class given as input to the training process from PyTorch Lightning and handle the random sampling with replacement of the training process, and the class that manage the different `DataClass` for the training, validation and test sets.\

#### Helpers files
• `helpers/init.py` : Contains the function `init` that initialize all the corresponding data, model and trainer regarding the arguments given by the user, and also initialize the WeightandBias logger to track the results.\
• `helpers/helpers.py` : Contains the function `reset_model` to reset all the weights of a model when required.\
• `helpers/pareto_fairness.py` : Contains the functions `compute_MMPF_size` and `compute_pareto_metrics` corresponding respectively to the computation of the `MMPF_size` metric presented in the report, and the computation of all the different variations of the MMPF metric which most of them are presented in the report.\
• `helpers/penalty_term.py` : Contains the function `compute_penalty_term` used in the context of the EDF training method where we presented various way of computing the penalty term added to the training loss.\
• `helpers/save_predictions.py` : Contains the function `save_predictions` to compute and save all the predictions made by a model depending of the training, validation, test sets or the various hyperparameters that defined the training of this model.\

#### Config files
• `config/get_args.py` : Contains the function `get_args` to initialize correctly the ArgumentParser regarding the inputs given in the scripts files by the user.\
• `config/info.py` : Contains generic informations about the datasets and the different protected attributes and their corresponding class indices.\


