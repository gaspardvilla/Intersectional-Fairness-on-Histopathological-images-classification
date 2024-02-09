from config.info import GENDERS, RACES
import pandas as pd
import numpy as np
import torch


def load_data(**kwargs):
    
    # Initialization of the references
    file_path = f'data/{kwargs["task"]}/{kwargs["task"]}_{kwargs["cancer"]}.csv'
    references = pd.read_csv(file_path)
    
    # Formating of the protected attributes
    references = _format_protected_attributes(references, 
                                        kwargs['age_format'], 
                                        protected_attributes = kwargs['protected_attributes'], 
                                        subgroups = kwargs['subgroups'])
    
    # Load the extracted features for each patient
    tensors_list = []
    response = torch.Tensor([])
    protected_attributes_values = torch.zeros((1, len(kwargs['protected_attributes'])), dtype = torch.int64)
    for idx in range(len(references)):
        tensor_features = torch.load(references.file.iloc[idx])
        response = torch.cat([response, torch.Tensor([references.label.iloc[idx]])])
        current_protected_att_val = torch.Tensor([list(references[kwargs['protected_attributes']].iloc[idx])])
        protected_attributes_values = torch.cat([protected_attributes_values, current_protected_att_val])
        if kwargs['add_protected_attributes']:
            tensor_features = torch.cat([tensor_features, current_protected_att_val.repeat(len(tensor_features), 1)], dim = 1)
        tensors_list += [tensor_features]
    
    # Transform the list into a nested tensor
    nested_data = torch.nested.nested_tensor(tensors_list)
    
    # Return the nested data set, the response and the protected attributes of each patient
    cols_to_return = ['subj', 'siteID', 'label'] + kwargs['protected_attributes']
    return nested_data, response, protected_attributes_values[1:], references[cols_to_return]
    
    
def _format_protected_attributes(references : pd.DataFrame, 
                                 age_format : str, 
                                 protected_attributes : list, 
                                 subgroups : list) -> pd.DataFrame:
    
    # Get the age of each patient regarding we want as continuous or discrete
    references['age_'] = ((pd.to_numeric(references.birth_days_to, errors = 'coerce').abs().fillna(-10000) / 365) + 0.5).astype(int)             # Transform the age in years - continuous case
    if age_format == 'discrete':
        references['age_'] = np.minimum((references.age_ / 10).astype(int), 9)                            # Transform the age in discrete variable
    elif age_format != 'continuous':
        raise ValueError(f'The variable age_format should be \'continuous\' or \'discrete\' instead of \'{age_format}\'.')
    
    # Transform the gender in discrete variable
    references['gender_'] = -1
    for c, gender in enumerate(GENDERS['str']):
        references.loc[references['gender'] == gender, 'gender_'] = GENDERS['class'][c]
    
    # Transform the race in discrete variable
    references['race_'] = -1
    for c, race in enumerate(RACES['str']):
        references.loc[references['race'] == race, 'race_'] = RACES['class'][c]
        
    # Remove the missing data and return the references
    references = references[references.age_ >= 0]
    references = references[references.race_ >= 0]
    references = references[references.gender_ >= 0]
    references.drop(columns = ['Unnamed: 0'], inplace = True)
    
    # Filtering by the subgroups given the protected attributes
    for idx, prot_att in enumerate(protected_attributes):
        references = references[references[prot_att].isin(subgroups[idx])]
    
    # Return the references
    references.reset_index(drop = True, inplace = True)
    return references