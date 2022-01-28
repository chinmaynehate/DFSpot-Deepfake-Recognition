import torch
from torch import nn as nn
from scipy.special import expit

class Ensemble(nn.Module):
    def __init__(self, models,device):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weightage = [1, 1, 1, 1]
        self.device = device

    def forward(self, x):
        scores = {}
        for i, model in enumerate(self.models):
            pred = model(x.to(self.device)).cpu().numpy().flatten()
            score = expit(pred.mean())
            scores[model.__class__.__name__] = score
        return scores

    

def ensemble(model_list,device):
    return Ensemble(model_list,device).eval().to(device)    