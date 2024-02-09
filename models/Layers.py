import torch.nn as nn
import torch


def MLP(in_features : int, fc_latent_size : int, 
        nb_classes : int, dropout : float):
    """ 
    This function builds MLP model with multiple layers with arguments the size of each
    layer (hidden or not) and the dropout argument for the training.

    Args:
        in_features (int): Input size
        fc_latent_size (int): Dimensions of the hidden layers
        nb_classes (int): Number of classes
        dropout (float, optional): dropout used during the training. Defaults to 0.5.

    Returns:
        nn.Sequential: Classifier model
    """
    
    # MLP
    if fc_latent_size is None:
        fc_latent_size = []
    fc_ins = [in_features] + fc_latent_size
    fc_outs = fc_latent_size + [nb_classes]
    
    # Replace final layer for fine tuning
    fc_list = []
    for i in range(len(fc_ins) - 1):
        fc_in = fc_ins[i]
        fc_out = fc_outs[i]
        
        fc = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(fc_in, fc_out),
            nn.ReLU()
        )
        
        fc_list.append(fc)
        
    fc_in = fc_ins[-1]
    fc_out = fc_outs[-1]
    
    fc = nn.Sequential(
        nn.Dropout(p = dropout),
        nn.Linear(fc_in, fc_out),
        nn.Sigmoid()
    )
    
    fc_list.append(fc)
    model = nn.Sequential(*fc_list)
    return model


class MILAttention(nn.Module):
    def __init__(self, featureLength : int, featureInside : int):
        '''
        Parameters:
            featureLength: Length of feature passed in from feature extractor(encoder)
            featureInside: Length of feature in MIL linear
        Output: tensor
            weight of the features
        '''
        super(MILAttention, self).__init__()
        self.featureLength = featureLength
        self.featureInside = featureInside

        self.attetion_V = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Tanh()
        )
        self.attetion_U = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Sigmoid()
        )
        self.attetion_weights = nn.Linear(self.featureInside, 1, bias = True)
        self.softmax = nn.Softmax(dim = 1)
        
        
    def forward(self, x: torch.Tensor):
        bz, pz, fz = x.shape
        x = x.view(bz*pz, fz)
        att_v = self.attetion_V(x)
        att_u = self.attetion_U(x)
        att = self.attetion_weights(att_u*att_v)
        weight = att.view(bz, pz, 1)
        
        weight = self.softmax(weight)
        weight = weight.view(bz, 1, pz)

        return weight