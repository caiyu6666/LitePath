import os
import csv
import time
import pickle

from datasets.Subtyping import Dataset_Subtyping, Dataset_Subtyping_Selected, Dataset_Subtyping_with_Att_Score
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
    # aps_results_dir = os.path.join(results_dir, "aps_weighted")

    print("[log dir] results directory: ", results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(aps_results_dir, exist_ok=True)

    # define dataset
    mil_dataset = Dataset_Subtyping_with_Att_Score(root=args.root, csv_file=args.csv_file, feature=args.feature, att_score_dir=aps_results_dir)

    # training and evaluation
    args.num_classes = mil_dataset.num_classes
    args.n_features = mil_dataset.n_features
    args.num_folds = mil_dataset.num_folds
    for fold in range(mil_dataset.num_folds):
        save_path = os.path.join(aps_results_dir, f"selection_results.pkl")
        # if os.path.exists(save_path):
        #     print(f"Selection results already exist at {save_path}. Skip ...")
        #     continue
        splits = mil_dataset.get_fold(fold)

        # 训练集：打乱
        mil_loaders = [DataLoader(mil_dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(splits[0]))]
        # 验证集和测试集：不打乱，顺序采样
        mil_loaders += [DataLoader(Subset(mil_dataset, split), batch_size=1, num_workers=4, pin_memory=True) for split in splits[1:3]]
        # 外部测试集：不打乱，顺序采样
        mil_external_loaders = {key: DataLoader(Subset(mil_dataset, split), batch_size=1, num_workers=4, pin_memory=True) for key, split in splits[3].items()} if splits[3] else None
        mil_loaders.append(mil_external_loaders)  # add external loaders if available

        #################################################
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            mil_model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))

        from models.APS.network import AdaPatchSelector
        from models.APS.engine import Engine
        engine = Engine(args, results_dir, fold, aps_results_dir)

        # infer & save att_score from the best mil model
        engine.infer_att_score(mil_model, mil_loaders)

        # define aps loaders
        aps_dataset = Dataset_APS(root=args.root, csv_file=args.csv_file, feature=args.feature, 
                              att_score_dir=aps_results_dir, aps_index=args.aps_index)
        args.n_shallow_features = aps_dataset.n_shallow_features
        aps_loaders = [DataLoader(aps_dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(split)) for split in splits[:3]]  # train, val, test
        aps_external_loaders = {key: DataLoader(aps_dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(split)) 
                            for key, split in splits[3].items()} if splits[3] else None
        aps_loaders.append(aps_external_loaders)  # add external loaders if available

        # define APS model
        aps_model = AdaPatchSelector(in_dim=args.n_shallow_features, out_dim=1)
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, aps_model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)

        # learning APS
        engine.learning(aps_model, aps_loaders, criterion, optimizer, scheduler)

        val_loader = DataLoader(Subset(mil_dataset, splits[1]), batch_size=1, num_workers=4, pin_memory=True)

        # # infer aps estimated att_score
        engine.infer_selection(aps_model, aps_loaders, score_only=True)

        selection_l = ["0_1000", "0_2000", "0_3000", "0_4000",
                       "500_0", "1000_0", "2000_0",
                       "50_950", "50_1950", "50_2950", "50_3950",
                       "100_900", "100_1900", "100_2900", "100_3900"]

        selection_results = {}
        for selection in selection_l:
            selection_results[selection], _ = engine.evaluate_with_selection(val_loader, mil_model, selection)
            print()

        with open(save_path, "wb") as f:
            pickle.dump(selection_results, f)
        print(f"Save selection results to {save_path}")



if __name__ == "__main__":
    args = aps_parse_args()
    results = main(args)
    print("finished!")
