from models.Baseline import Baseline
from models.Martinez import Martinez
from models.Foulds import Foulds
from models.Diana import Diana


def reset_model(model):
    if model.name() == 'Baseline': model_return = Baseline(model.in_features, **model.kwargs)
    elif model.name() == 'Martinez': model_return = Martinez(model.in_features, **model.kwargs)
    elif model.name() == 'Foulds': model_return = Foulds(model.in_features, **model.kwargs)
    elif model.name() == 'Diana': model_return = Diana(model.in_features, **model.kwargs)
    return model_return