from config.info import AGES, RACES, GENDERS, METRICS
import sklearn.metrics as sm
import torchmetrics as tm
import pandas as pd
import numpy as np
import torch


def compute_metric(df : pd.DataFrame, metric : str = None, **kwargs):
    # Initialization
    metrics_df = compute_metrics(df, **kwargs)
    metric_df = None
    
    # Return the dataframe
    # In case there is no metric precised we get all the metrics
    if metric is None: return metrics_df
    else: return metric_df


def compute_metrics(df : pd.DataFrame, **kwargs):
    # Initialization
    metrics_df = _init_metrics_df(kwargs['protected_attributes'])

    # Compute the metrics for each protected_attributes
    for age in AGES['str']:
        for race in RACES['str']:
            for gender in GENDERS['str']:
                
                # Keep only the data corresponding to the subgroup
                indices = df[(df.age_str == age) & (df.race_str == race) & (df.gender_str == gender)].index
                sub_df = df.iloc[indices]
                
                # Check if the dataframe is empty
                if sub_df.empty: add_df = pd.DataFrame([[age, race, gender, 0] + [np.nan] * len(METRICS)], columns = metrics_df.columns)
                else: add_df = test_compute_metrics(sub_df, kwargs['protected_attributes'])
                
                # Update the dataframe
                metrics_df = pd.concat([metrics_df, add_df])
    
    # Reset the indices and return the metrics
    metrics_df.reset_index(drop = True, inplace = True)
    return metrics_df[metrics_df['size'] > 0]
                
                
def _init_metrics_df(protected_attributes, subgroup_data : list = None):
    # Initialization
    if subgroup_data is None: metrics_df = pd.DataFrame(columns = protected_attributes + ['size'] + METRICS)
    else: metrics_df = pd.DataFrame(data = [subgroup_data + [np.nan] * (len(METRICS) + 1)],
                                    columns = protected_attributes + ['size'] + METRICS)
    return metrics_df


def test_compute_metrics(df : pd.DataFrame, protected_attributes : list):
    # Initialization
    subgroups_data = []
    for pa in protected_attributes: subgroups_data += [df[pa].iloc[0]]
    metrics_df = _init_metrics_df(protected_attributes, subgroup_data = subgroups_data)
    
    # Pre-compute metrics
    TP = len(df[(df['pred'] == 1) & (df['label'] == 1)])
    TN = len(df[(df['pred'] == 0) & (df['label'] == 0)])
    FP = len(df[(df['pred'] == 1) & (df['label'] == 0)])
    FN = len(df[(df['pred'] == 0) & (df['label'] == 1)])
    
    # Computation of all the metrics
    metrics_df.at[0, 'size'] = len(df)
    
    metrics_df.at[0, 'TP'] = TP
    metrics_df.at[0, 'TN'] = TN
    metrics_df.at[0, 'FP'] = FP
    metrics_df.at[0, 'FN'] = FN
    metrics_df.at[0, 'F1-score'] = tm.F1Score(task = 'multiclass', num_classes = 2)(torch.Tensor(list(df.pred)), torch.Tensor(list(df.label))).item()
    try: metrics_df.at[0, 'ACC'] = (TN + TP) / (TN + TP + FN + FP) 
    except: pass
    
    metrics_df.at[0, 'AUROC'] = tm.AUROC(task = 'multiclass', num_classes = 2)(torch.from_numpy(np.reshape(list(df.pred_raw), (len(df), 2))), torch.from_numpy(np.array(df.label))).item()
    try: metrics_df.at[0, 'AUC'] = sm.roc_auc_score(torch.Tensor(list(df.label)), torch.Tensor(list(df.pred)))
    except: pass
    
    try: metrics_df.at[0, 'TPR - Recall'] = TP / (TP + FN)
    except: pass
    try: metrics_df.at[0, 'PPV - Precision'] = TP / (TP + FP)
    except: pass
    try: metrics_df.at[0, 'TNR'] = TN / (TN + FP)
    except: pass
    try: metrics_df.at[0, 'FPR'] = FP / (FP + TN)
    except: pass
    try: metrics_df.at[0, 'FNR'] = FN / (FN + TP)
    except: pass
    try: metrics_df.at[0, 'FDR'] = FP / (FP + TP)
    except: pass
    try: metrics_df.at[0, 'FOR'] = FN / (FN + TN)
    except: pass
    try: metrics_df.at[0, 'NPV'] = TN / (TN + FN)
    except: pass
    try: metrics_df.at[0, 'RPP'] = (FP + TP) / (TN + TP + FN + FP)
    except: pass
    try: metrics_df.at[0, 'RNP'] = (FN + TN) / (TN + TP + FN + FP)
    except: pass

    # Return the metrics
    return metrics_df