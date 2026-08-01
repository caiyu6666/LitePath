import os
import csv
import time

from datasets.Subtyping import Dataset_Subtyping, Dataset_Subtyping_Selected
from datasets.APS import Dataset_APS
from utils.options import aps_parse_args
from utils.util import set_seed, CV_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    results_dir = "./results/{study}/{feature}/{model}_seed_{seed}".format(
        seed=args.seed,
        study=args.study,
        model=args.model,
        feature=args.feature,
    )
    aps_results_dir = os.path.join(results_dir, "aps")
    # topk_results_dir = os.path.join(results_dir, "topk")
    uniformk_results_dir = os.path.join(results_dir, "uniformk")

    print("[log dir] results directory: ", results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(aps_results_dir, exist_ok=True)
    # os.makedirs(topk_results_dir, exist_ok=True)
    os.makedirs(uniformk_results_dir, exist_ok=True)

    # define dataset
    mil_dataset = Dataset_Subtyping(root=args.root, csv_file=args.csv_file, feature=args.feature)

    # training and evaluation
    args.num_classes = mil_dataset.num_classes
    args.n_features = mil_dataset.n_features
    args.num_folds = mil_dataset.num_folds
    for fold in range(mil_dataset.num_folds):
        splits = mil_dataset.get_fold(fold)
        # mil loaders

        # val_loader = DataLoader(mil_dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(splits[1]))
        # loaders += [DataLoader(Subset(dataset, split), batch_size=1, num_workers=4, pin_memory=True) for split in splits[1:3]]
        mil_loaders = [DataLoader(Subset(mil_dataset, splits[i]), batch_size=1, num_workers=4, pin_memory=True) for i in range(1, 3)]  # val, test
        mil_external_loaders = {key: DataLoader(Subset(mil_dataset, split), batch_size=1, num_workers=4, pin_memory=True) 
                            for key, split in splits[3].items()} if splits[3] else None
        mil_loaders.append(mil_external_loaders)  # add external loaders if available

        #################################################
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            mil_model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))

        from models.APS.engine import Engine
        engine = Engine(args, results_dir, fold, aps_results_dir)
        k_list = [50, 100, 200, 300, 500, 1000, 2000, 3000]
        
        engine.evaluate_topk(mil_model, mil_loaders[1], k_list=k_list, save_dir=uniformk_results_dir, mode='uniform', key='test')
        engine.evaluate_topk(mil_model, mil_loaders[1], k_list=k_list, save_dir=uniformk_results_dir, mode='top', key='test')


if __name__ == "__main__":
    args = aps_parse_args()
    results = main(args)
    print("finished!")
