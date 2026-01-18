import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import yaml
import shutil
import time

from models import get_model
from utils import build_optimizer, build_lr_scheduler, build_dataloader_fast, save_checkpoint, load_checkpoint, AverageMeter
from utils import print_network, find_latest_checkpoint
from losses import DistillationLossMulti
import numpy as np
import random


def init_seeds(rank, seed=42):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    # slower, but more reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_worker(rank, args, config):

    torch.cuda.set_device(rank)
    if config['world_size'] > 1:
        dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'], world_size=config['world_size'], rank=rank)

    model_config = config['model']
    student = get_model(model_config['student']['name'], rank, **model_config['student'],
                        teacher_dict=model_config['teacher'])

    if config['world_size'] > 1:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[rank])

    student.train()

    if rank == 0:
        print("=========== Student ===========")
        print_network(student, architecture=True)

    criterion = DistillationLossMulti(config)
    optimizer = build_optimizer(student, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)

    iteration = 0
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(config['save_dir'])
        print(f"=> Resuming from checkpoint: {latest_checkpoint}")
        iteration = load_checkpoint(latest_checkpoint, student, optimizer, lr_scheduler)
    else:
        print("Starting training from scratch.")

    dataloader = build_dataloader_fast(config, iteration=iteration)

    max_iterations = config['max_iterations']

    if rank == 0:
        if args.resume:
            print(f"Resuming wandb log {args.wandb_id}")
            wandb_logger = wandb.init(project=config['project_name'], name=config['name'],
                                      settings=wandb.Settings(console="off"), resume="must", id=args.wandb_id)
        else:
            wandb_logger = wandb.init(project=config['project_name'], name=config['name'],
                                      settings=wandb.Settings(console="off"))
            wandb_logger.config.update(config)
    else:
        wandb_logger = None

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_dict = {k: AverageMeter() for k in model_config['teacher'].keys()}

    scaler = torch.cuda.amp.GradScaler()
    grad_clip = config['optimizer'].get('grad_clip', 1.0)
    start_time = time.time()
    while iteration < max_iterations:
        for data_batch in dataloader:
            if iteration >= max_iterations:
                break

            img = data_batch['image'].cuda()
            teacher_features = {k: v.cuda() for k, v in data_batch['teacher_features'].items()}
            bs = img.size(0)

            data_time.update(time.time() - start_time)

            with torch.cuda.amp.autocast():
                # student_features = projector(student(img))
                student_features = student(img)
                loss, loss_dict = criterion(student_features, teacher_features)

                losses.update(loss.item(), bs)
                for k, v in loss_dict.items():
                    losses_dict[k].update(v.item(), bs)

            scaler.scale(loss).backward()
            lr_scheduler.step(iteration)
            # Gradient clipping
            scaler.unscale_(optimizer)
            # grad_norm_before = get_grad_norm(student.parameters())
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
            #

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            iteration += 1

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if (iteration == 1 or iteration % config['print_freq'] == 0) and rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iteration {iteration}:\t"
                      f"Batch_Time = {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      f"Data_Time = {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      f"LR = {current_lr:.6f}\t"
                      f"Loss = {losses.val:.4f} ({losses.avg:.3f})\t"
                      # f"Grad_Norm = {grad_norm_before:.4f}"
                      )
                log_dict = {
                    "Loss": losses.val,
                    "Learning Rate": current_lr,
                    "Batch Time": batch_time.val,
                    "Data Time": data_time.val,
                    # "Grad Norm": grad_norm_before
                }
                losses_val_dict = {k: v.val for k, v in losses_dict.items()}
                print(losses_val_dict)

                log_dict.update(losses_val_dict)
                wandb_logger.log(log_dict, step=iteration)

            # Save checkpoint periodically
            if iteration % config['save_freq'] == 0 and rank == 0:
                # ckpt_name = f"{model_config['student']['name']}_{iteration}.pth"
                ckpt_name = "checkpoint.pth"
                save_checkpoint({
                    'iteration': iteration,
                    'state_dict': student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, config['save_dir'], ckpt_name)
                with open(os.path.join(config['save_dir'], 'latest.txt'), 'w') as f:
                    f.write(str(iteration))

        # Rebuild the dataloader if the sampler has a start index greater than 0
        if dataloader.sampler.start_index > 0:
            print("=> Rebuilding dataloader for next epoch...")
            dataloader = build_dataloader_fast(config)

    if rank == 0:
        save_checkpoint(student.module.state_dict(), config['save_dir'], 'final.pth')
        wandb_logger.finish()

    if dist.is_initialized():
        dist.destroy_process_group()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature Distillation Training")
    parser.add_argument('--config', type=str, default='configs/exp1.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint.')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB run ID for resuming the run.')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    args = parse_arguments()
    config = load_config(args.config)

    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    config['name'] = exp_name
    config['save_dir'] = os.path.join(config['save_root'], exp_name)
    os.makedirs(config['save_dir'], exist_ok=True)
    shutil.copy(args.config, config['save_dir'])

    if config['world_size'] > 1:
        mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(args, config))
    else:
        main_worker(config['gpu'], args, config)


if __name__ == "__main__":
    main()
