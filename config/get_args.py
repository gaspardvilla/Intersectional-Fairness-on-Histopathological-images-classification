from config.info import COMBS_BASELINE, COMBS_FOULDS, COMBS_MARTINEZ, AGES, RACES, GENDERS
import argparse


def get_args(args : argparse.Namespace = None):
    # Initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help = 'Seed', default = 194825, type = int)
    parser.add_argument('--allow_wandb', help = 'Allow WandB tracking', default = 1, type = int)
    parser.add_argument('--ID', help = 'ID of the submission', default = 0, type = int)
    parser.add_argument('--job_id', help = 'Run id in case of multiple jobs submitted', default = -1, type = int)
    
    # Data
    parser.add_argument('--task', help = 'tumor_detection or cancer_classification', default = 'cancer_classification', type = str)
    parser.add_argument('--cancer', help = 'Cancer types to be detected or classified (check data set)', default = 'luad_lusc_FS', type = str)
    parser.add_argument('--age_format', help = 'Indicates the format of the age_ protected attribute', default = 'discrete', type = str)
    parser.add_argument('--custom_subgroups', help = 'Indicates if we consider only some specific subgroups', default = 0, type = int)
    parser.add_argument('--add_protected_attributes', help = 'Add the protected attributes to the features', default = 1, type = int)
    parser.add_argument('--nb_classes', help = 'Number of classes for the prediction', default = 2, type = int)

    # Model or method
    parser.add_argument('--model', help = 'Fairness method used : [Baseline, Foulds]', default = 'Foulds', type = str)
    parser.add_argument('--eps_1', help = 'Foulds method - epsilon_1 value to fit differently the data', default = 0.0, type = float)
    parser.add_argument('--save_model', help = 'Save the model', default = 1, type = int)
    parser.add_argument('--ckpt_path', help = 'Checkpoint path where to save the model', default = 'results/ckpt', type = str)
    parser.add_argument('--pt_method', help = 'Penalty term method - Foulds method only', default = 'DF_pos', type = str)
    parser.add_argument('--lambda_', help = 'lambda value for the penalty term', default = 0.01, type = float)
    parser.add_argument('--MIL_latent_size', help = 'MIL_latent_size - hyperparameter of the model', default = 256, type = int)
    parser.add_argument('--fc_latent_size', help = 'fc_latent_size - hyperparameter of the model', default = [20, 20], type = list)
    parser.add_argument('--alpha_', help = 'alpha value for Martinez method', default = 0.5, type = float)

    # Training
    parser.add_argument('--nb_epochs', help = 'Maximum nb of epochs', default = 500, type = int)
    parser.add_argument('--dropout', help = 'dropout', default = 0.5, type = float)
    parser.add_argument('--check_val', help = 'Check validation every n epochs', default = 10, type = int)
    parser.add_argument('--split_validation', help = 'Indicates if we give a validation set during the split', default = 1, type = int)
    parser.add_argument('--split_regarding_subgroups', help = 'Split the train/val/test sets by keeping all subgroups rationnaly present in all sets', default = 1, type = int)
    parser.add_argument('--split_ratio', help = 'Ratio used to split the data set into train / val / test sets', default = 0.2, type = float)
    parser.add_argument('--NB_STEPS', help = 'Nb steps for Martinez / Diana methods', default = 100, type = int)

    # Testing
    parser.add_argument('--save_preds', help = 'Save the predictions made by the model', default = 1, type = int)
    args = parser.parse_args(namespace = args)
    
    # Clear all the arguments
    args.run_path = ''
    if args.job_id >= 0: args = _multi_jobs(args)
    args = _clear_args(args)
    
    # Build the ckpt_path
    args.ckpt_path = f'{args.ckpt_path}/{args.run_path}/checkpoint.ckpt'
    
    # Return the arguments
    return args


def _clear_args(args):
    # In case we want to work on custom subgroups (or not)
    if args.custom_subgroups :
        args.protected_attributes = ['race_', 'gender_']
        args.subgroups = [[1, 3],
                          [0, 1]]
    else:
        args.protected_attributes = ['age_', 'race_', 'gender_']
        args.subgroups = [AGES['class'],
                          RACES['class'],
                          GENDERS['class']]
    
    # Initialization of the path of the run
    args.run_path = f'run_{args.ID}/add_protected_atts_{int(args.add_protected_attributes)}/{args.model}/{args.task}/{args.cancer}{args.run_path}'
    return args


def _multi_jobs(args):
    # In case we submit multiple jobs at the same time
    if (args.model == 'Baseline') or (args.model == 'Diana') or (args.model == 'Martinez'):
        comb = COMBS_BASELINE[args.job_id]
        args.task = comb[0]
        args.cancer = comb[1]
    elif args.model == 'Foulds':
        comb = COMBS_FOULDS[args.job_id]
        args.task = comb[0]
        args.cancer = comb[1]
        args.lambda_ = comb[2]
        args.pt_method = comb[3]
        args.run_path = f'/lambda_{args.lambda_}/{args.pt_method}'
    return args