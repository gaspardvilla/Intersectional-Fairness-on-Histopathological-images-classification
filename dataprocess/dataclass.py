from dataprocess.dataloader import load_data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.utils import resample
import pandas as pd
import numpy as np
import random
import torch


class Data():
    def __init__(self, **kwargs):
        # Initialization
        super().__init__()
        self.file = f'{kwargs["task"]}_{kwargs["cancer"]}.csv'
        self.protected_attributes = kwargs['protected_attributes']
        self.subgroups = kwargs['subgroups']
        self.data, self.response, self.protected_attribbutes_values, self.references = load_data(**kwargs)
        self.ratio = kwargs['split_ratio']
        self.split_validation = kwargs['split_validation']
        self.split_regarding_subgroups = kwargs['split_regarding_subgroups']
        self.nb_features = self.data[0].shape[-1]
        self.nb_subgroups = len(np.unique(self.references[self.protected_attributes].values, axis = 0))
        self.nb_workers = 8
        
        # Split the data set
        self._split_subgroups()
        
    # def split(self):
    #     # Initialization
    #     n_test = int(len(self.references) * self.ratio)
    #     if self.split_regarding_subgroups: self._split_subgroups()
    #     else: self._split_random(n_test)
    
    def _split_subgroups(self):
        # Initialization
        train_subj_list = []
        val_subj_list = []
        test_subj_list = []
        train_distribution = [0, 0]
        
        # Get the values of unique subgroups prensent in the data set
        unique_subgroups = np.unique(self.references[self.protected_attributes].values, axis = 0)

        # Loop on all the unique subgroups in the references
        for subgroup in unique_subgroups:
            
            # Extract the references that are in this subgroup only
            cond = self.references[self.protected_attributes[0]] == subgroup[0]
            for idx, att in enumerate(self.protected_attributes[1:]):
                cond = cond & (self.references[att] == subgroup[idx+1])
                
            # Do the following steps with the negative and positive labels
            for label in range(2):
                
                # Restrict the focus on the current class
                cond_l = cond & (self.references.label == label)
                subjects = list(self.references[cond_l].subj)
                
                # Shuffle and split it between train / validation / test sets
                random.shuffle(subjects)
                if len(subjects) == 1: 
                    train_subj_list += [subjects[0]]
                    train_distribution[label] = train_distribution[label] + 1
                elif len(subjects) == 2: 
                    train_subj_list += [subjects[0]]
                    val_subj_list += [subjects[1]]
                    train_distribution[label] = train_distribution[label] + 1
                elif len(subjects) == 3:
                    train_subj_list += [subjects[0]]
                    val_subj_list += [subjects[1]]
                    test_subj_list += [subjects[2]]
                    train_distribution[label] = train_distribution[label] + 1
                elif len(subjects) == 4:
                    train_subj_list += [subjects[0], subjects[1]]
                    val_subj_list += [subjects[2]]
                    test_subj_list += [subjects[3]]
                    train_distribution[label] = train_distribution[label] + 2
                else:
                    sub_n = int(self.ratio * len(subjects))
                    test_subj_list += subjects[: sub_n]
                    val_subj_list += subjects[sub_n : 2 * sub_n]
                    train_subj_list += subjects[2 * sub_n :]
                    train_distribution[label] = train_distribution[label] + len(subjects[2 * sub_n :])
        
        # Set the weights for the train distribution among the two classes 0 and 1
        N_train_dist = sum(train_distribution)
        self.train_weights = torch.Tensor([N_train_dist / train_distribution[0], N_train_dist / train_distribution[1]])

        if self.split_validation:
            # Create the data classes
            self.TrainSet = DataClass(self.data, self.response, self.references, 
                                        subjects = train_subj_list, resampling = True,
                                        protected_attributes = self.protected_attributes)
            self.ValidationSet = DataClass(self.data, self.response, self.references, 
                                            subjects = val_subj_list, resampling = False,
                                            protected_attributes = self.protected_attributes)
            self.TestSet = DataClass(self.data, self.response, self.references, 
                                        subjects = test_subj_list, resampling = False,
                                        protected_attributes = self.protected_attributes)

            # Loaders
            self.TrainLoader = DataLoader(self.TrainSet, batch_size = len(self.TrainSet), shuffle = True, drop_last = False)
            self.ValidationLoader = DataLoader(self.ValidationSet, batch_size = 1, shuffle = False, drop_last = False, num_workers = self.nb_workers)
            self.TestLoader = DataLoader(self.TestSet, batch_size = 1, shuffle = False, drop_last = False, num_workers = self.nb_workers)
        
        else:
            # Create the data classes
            self.TrainSet = DataClass(self.data, self.response, self.references, 
                                        subjects = train_subj_list + val_subj_list, resampling = True,
                                        protected_attributes = self.protected_attributes)
            self.TestSet = DataClass(self.data, self.response, self.references, 
                                        subjects = test_subj_list, resampling = False,
                                        protected_attributes = self.protected_attributes)

            # Loaders
            self.TrainLoader = DataLoader(self.TrainSet, batch_size = len(self.TrainSet), shuffle = True, drop_last = False)
            self.TestLoader = DataLoader(self.TestSet, batch_size = 1, shuffle = False, drop_last = False, num_workers = self.nb_workers)

    # def _split_random(self, n_test):
    #     # Extract the subjects list
    #     subj_list = list(self.references.subj)
    #     random.shuffle(subj_list)
        
    #     # Get the Train, Test and Validation data classes and loaders
    #     if self.split_validation:
    #         # Classes
    #         self.TrainSet = DataClass(self.data, self.response, self.references, 
    #                                   subjects = subj_list[2 * n_test :], resampling = True,
    #                                   protected_attributes = self.protected_attributes)
    #         self.ValidationSet = DataClass(self.data, self.response, self.references, 
    #                                        subjects = subj_list[n_test : 2 * n_test], resampling = False,
    #                                        protected_attributes = self.protected_attributes)
    #         self.TestSet = DataClass(self.data, self.response, self.references, 
    #                                  subjects = subj_list[: n_test], resampling = False,
    #                                  protected_attributes = self.protected_attributes)
            
    #         # Loaders
    #         self.TrainLoader = DataLoader(self.TrainSet, batch_size = len(self.TrainSet), shuffle = True, drop_last = False)
    #         self.ValidationLoader = DataLoader(self.ValidationSet, batch_size = 1, shuffle = False, drop_last = False)
    #         self.TestLoader = DataLoader(self.TestSet, batch_size = 1, shuffle = False, drop_last = False)
        
    #     # Get the Train and Test data classes and loaders
    #     else:
    #         # Classes
    #         self.TrainSet = DataClass(self.data, self.response, self.references, 
    #                                   subjects = subj_list[n_test :], resampling = True,
    #                                   protected_attributes = self.protected_attributes)
    #         self.TestSet = DataClass(self.data, self.response, self.references, 
    #                                  subjects = subj_list[: n_test], resampling = False,
    #                                  protected_attributes = self.protected_attributes)
            
    #         # Loaders
    #         self.TrainLoader = DataLoader(self.TrainSet, batch_size = len(self.TrainSet), shuffle = True, drop_last = False)
    #         self.TestLoader = DataLoader(self.TestSet, batch_size = 1, shuffle = False, drop_last = False)
            
    def __str__(self):
        # Print the composition of the data set
        print('[ --------------- GENERAL --------------- ]')
        print(f'File name : {self.file}')
        print(f'Total nb of data : {len(self.references)}')
        print(f'Nb of features : {self.nb_features}')
        print(f'Split ratio : {self.ratio}')
        print(f'Validation set : {self.split_validation}')
        print('')
        
        # Specifications
        print(f'Train set size : {len(self.TrainSet)}')
        if self.split_validation: print(f'Validation set size : {len(self.ValidationSet)}')
        else: print(f'Validation set size : None')
        print(f'Test set size : {len(self.TestSet)}')
        
        return ''
        


class DataClass(Dataset):
    def __init__(self, data : torch.nested.nested_tensor, 
                 response : torch.Tensor, 
                 references : pd.DataFrame, 
                 subjects : list, 
                 resampling : bool, 
                 protected_attributes : list):
        # Initialization
        super().__init__()
        self.data = []
        self.response = []
        self.references = pd.DataFrame()
        self.resampling = resampling
        self.protected_attributes = protected_attributes
        
        # Extract the data of the subjects
        if subjects is None : subjects = list(references.subj)
        for subj in subjects:
            
            # Get the index location of the subject and upadte the data and response
            idx_subj = references.index[references.subj == subj].tolist()[0]
            self.data += [data[idx_subj]]
            self.response += [response[idx_subj]]
            
            # Update the references
            data_dict = {'subj': [references.iloc[idx_subj].subj],
                         'siteID': [references.iloc[idx_subj].siteID], 
                         'label': [references.iloc[idx_subj].label]}
            for prot_att in self.protected_attributes:
                data_dict.update({prot_att : [references.iloc[idx_subj][prot_att]]})
            current_ref = pd.DataFrame(data_dict)
            self.references = pd.concat((self.references, current_ref), 
                                         axis = 0, ignore_index = True)

    def __len__(self) -> int:
        """ Length of the data set. """
        return len(self.response)

    def __getitem__(self, index) -> tuple:
        """ Returns the data item at the location given by the index. """
        # Extract data and response
        data = self.data[index]
        resp = self.response[index]
        att = torch.Tensor(list(self.references.iloc[index][self.protected_attributes]))
        
        # Random resampling of the data with replacement
        if self.resampling: data_r = resample(data, replace = True, n_samples = 200)
        else: data_r = data

        # Formatting of the data
        resp = resp.unsqueeze(0)
        resp = torch.cat((1 - resp, resp), 0)
        return data_r, resp, att