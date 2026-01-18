import torch
import timm
import numpy as np
from torchvision import transforms
import re
from .mil_model import DAttention, AdaPatchSelector

__implemented_models = {
    'LiteFM-S': 'ckpts/LiteFM-S.pth',
    'LiteFM': 'ckpts/LiteFM.pth',
    'LiteVirchow2': 'ckpts/LiteVirchow2.pth',
    'LiteFM-L': 'ckpts/LiteFM-L.pth',
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model(model_name, device):
    if model_name == 'LiteFM-S':
        from .litefm import custom_vit_tiny_patch16_224
        model = custom_vit_tiny_patch16_224(device, __implemented_models['LiteFM-S'], proj_dim=1024, out_dim_dict=None)

    elif model_name == 'LiteFM':
        from .litefm import custom_vit_small_patch16_224
        model = custom_vit_small_patch16_224(device, __implemented_models['LiteFM'], proj_dim=1024, out_dim_dict=None)

    elif model_name == 'LiteFM-L':
        from .litefm import custom_vit_base_patch16_224
        model = custom_vit_base_patch16_224(device, __implemented_models['LiteFM-L'], proj_dim=1024, out_dim_dict=None)

    elif model_name == 'LiteVirchow2':
        from .litefm import custom_vit_small_patch16_224
        model = custom_vit_small_patch16_224(device, __implemented_models['LiteVirchow2'], proj_dim=1024, out_dim_dict=None)

    else:
        raise NotImplementedError(f'{model_name} is not implemented')

    model = model.eval()

    return model


def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if 'litefm' in model_name.lower():
        from .litefm import get_litefm_trans
        custom_trans = get_litefm_trans()

    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
