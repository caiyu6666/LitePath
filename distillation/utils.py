import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import re

from timm.scheduler import CosineLRScheduler
from datasets import FastPathologyDataset, ResumeDistributedSampler
from collections import OrderedDict


def build_transform(transform_config):
    if transform_config['custom'] is None:
        transform_items = transform_config['default']
        transform_list = []
        for key, value in transform_items.items():
            if key == 'resize':
                transform_list.append(transforms.Resize((value['input_size'], value['input_size'])))
            elif key == 'random_resized_crop':
                transform_list.append(
                    transforms.RandomResizedCrop(
                        size=(value['input_size'], value['input_size']),
                        scale=value['scale'], ratio=value['ratio'],
                        interpolation=3
                    )
                )
            elif key == 'random_horizontal_flip':
                transform_list.append(transforms.RandomHorizontalFlip(p=value))
            elif key == 'random_vertical_flip':
                transform_list.append(transforms.RandomVerticalFlip(p=value))
            elif key == 'color_jitter':
                transform_list.append(
                    transforms.ColorJitter(
                    brightness=value['brightness'],
                    contrast=value['contrast'],
                    saturation=value['saturation'],
                    hue=value['hue']
                    )
                )
            elif key == 'normalize':
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=tuple(value['mean']), std=tuple(value['std'])))
            else:
                raise ValueError('Unknown transform key: {}'.format(key))
        return transforms.Compose(transform_list)

    elif transform_config['custom'] == 'virchow2':
        from models.virchow2 import get_virchow_trans
        return get_virchow_trans()

    else:
        raise NotImplementedError('Custom transform {} is not implemented.'.format(transform_config['custom']))

def build_optimizer(student, config):
    print("=> Building optimizer ... ")
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=optimizer_config['learning_rate'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    return optimizer


def build_lr_scheduler(optimizer, config):
    print("=> Building learning rate scheduler ... ")
    lr_scheduler_config = config['lr_scheduler']
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=lr_scheduler_config['t_initial'],  # 总步数或 epoch 数
        lr_min=lr_scheduler_config['lr_min'],  # 最小学习率
        warmup_lr_init=lr_scheduler_config['warmup_lr'],  # 预热初始学习率
        warmup_t=lr_scheduler_config['warmup_steps'],  # 预热步数
    )
    return scheduler


def build_dataloader_fast(config, iteration=0):
    dataloader_config = config['dataloader']
    transform_config = config['transform']

    print("=> Building transform ... ")
    transform = build_transform(transform_config)

    print("=> Building dataloader ... ")
    dataset = FastPathologyDataset(root=dataloader_config['data_root'], transform=transform,
                                   teachers=list(config['model']['teacher'].keys()))

    batch_size = dataloader_config['batch_size']
    sampler = ResumeDistributedSampler(dataset, start_index=iteration*batch_size) if config['world_size'] > 1 else None

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            pin_memory=True, shuffle=(sampler is None), num_workers=dataloader_config['num_workers'],
                            persistent_workers=True)
    return dataloader


def save_checkpoint(state, output_dir, filename="checkpoint_last.pth"):
    """Save training state to a checkpoint file."""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    print("=> Saved checkpoint '{}'".format(filepath))


def load_checkpoint(filepath, student, optimizer, lr_scheduler, projector=None):
    """Load training state from a checkpoint file."""
    checkpoint = torch.load(filepath, map_location="cuda")

    state_dict = checkpoint['state_dict']
    state_dict = adjust_state_dict_ddp(state_dict)
    student.load_state_dict(state_dict)
    if projector is not None:
        projector_state_dict = checkpoint['projector']
        projector_state_dict = adjust_state_dict_ddp(projector_state_dict)
        projector.load_state_dict(projector_state_dict)

    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return checkpoint['iteration']


def adjust_state_dict_ddp(state_dict):
    adjusted_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k if k.startswith("module.") else f"module.{k}"
        adjusted_state_dict[new_key] = v
    return adjusted_state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def print_network(model, architecture=False):
    num_params = count_parameters(model)
    num_params_str = f"{num_params / 1e6:.2f}M" if num_params > 1e6 else f"{num_params / 1e3:.2f}K"
    print(f"Number of parameters: {num_params_str}")
    if architecture:
        print("Model architecture:")
        print(model)


def find_latest_checkpoint(directory):
    """
    Find the file with the largest number at the end (e.g., _1234.pth) in the specified directory.

    Args:
        directory (str): The directory to search for checkpoint files.

    Returns:
        str: The path to the latest checkpoint file. If no matching file is found, returns None.
    """
    file_l = os.listdir(directory)
    if "checkpoint.pth" in file_l:
        return os.path.join(directory, "checkpoint.pth")
    else:
        max_number = -1
        latest_checkpoint = None

        # Regular expression to match filenames ending with "_number.pth"
        regex = re.compile(r".*_(\d+)\.pth$")

        # Iterate over all files in the directory
        for file in file_l:
            match = regex.match(file)
            if match:
                number = int(match.group(1))  # Extract the number from the filename
                if number > max_number:
                    max_number = number
                    latest_checkpoint = os.path.join(directory, file)

        return latest_checkpoint


def get_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
