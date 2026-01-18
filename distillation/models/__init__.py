import torch
import numpy as np


def get_model(model_name, device, **kwargs):
    if model_name == 'LiteFM-S':
        from .litefm import custom_vit_tiny_patch16_224
        model = custom_vit_tiny_patch16_224(pretrained=False, proj_dim=kwargs['proj_dim'],
                                            out_dim_dict={k: v['output_dim'] for k, v in kwargs['teacher_dict'].items()}).to(device)

    elif model_name == 'LiteFM' or model_name == 'LiteVirchow2':
        from .litefm import custom_vit_small_patch16_224
        model = custom_vit_small_patch16_224(pretrained=False, proj_dim=kwargs['proj_dim'],
                                            out_dim_dict={k: v['output_dim'] for k, v in kwargs['teacher_dict'].items()}).to(device)

    elif model_name == 'LiteFM-L':
        from .litefm import custom_vit_base_patch16_224
        model = custom_vit_base_patch16_224(pretrained=False, proj_dim=kwargs['proj_dim'],
                                            out_dim_dict={k: v['output_dim'] for k, v in kwargs['teacher_dict'].items()}).to(device)
    
    elif model_name == 'virchow2':
        from .virchow2 import get_virchow_model
        model = get_virchow_model(device)

    elif model_name.lower() == 'h-optimus-1':
        from .h_optimus_1 import get_model
        model = get_model(device)
    
    elif model_name.lower() == 'uni2':
        from .uni2 import get_uni_model
        model = get_uni_model(device)

    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    return model


def get_custom_transformer(model_name):
    if model_name.lower() == 'virchow2':
        from .virchow2 import get_virchow_trans
        custom_trans = get_virchow_trans()

    elif model_name.lower() == 'h-optimus-1':
        from .h_optimus_1 import get_hoptimus_trans
        custom_trans = get_hoptimus_trans()

    elif model_name.lower() == 'uni2':
        from .uni2 import get_uni_trans
        custom_trans = get_uni_trans()

    elif 'litefm' in model_name.lower():
        from .litefm import get_litefm_trans
        custom_trans = get_litefm_trans()

    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans
