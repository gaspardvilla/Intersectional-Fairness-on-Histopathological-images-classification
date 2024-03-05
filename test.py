from helpers.pareto_fairness import compute_pareto_metrics
from config.info import AGES, RACES, GENDERS, COMBS_BASELINE, COMBS_FOULDS, COMBS_MARTINEZ
from visualization.subgroup_distribution import plot_dist
from dataprocess.dataloader import load_data
from dataprocess.dataclass import Data
from config.get_args import get_args
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from lightning import seed_everything
from plotly.subplots import make_subplots
from helpers.init import init
import numpy as np
import argparse
import pandas as pd
import random
import torch.nn as nn
import torch
import torchmetrics as tm
import pickle

loss_fct = nn.CrossEntropyLoss()
protected_atts = ['age_', 'race_', 'gender_']


if __name__ == '__main__':
    
    # args
    args = argparse.Namespace()
    args.model = 'Baseline' 
    args.ID = 1010
    args.add_protected_attributes = 1
    args.allow_wandb = False
    args.job_id = 9
    args = get_args(args = args)
    args.fc_latent_size = [20, 20]

    # Load the data, the model, wandb and the trainer
    seed_everything(args.seed)
    data, model, _, _ = init(**vars(args))
    
    print('yolo')