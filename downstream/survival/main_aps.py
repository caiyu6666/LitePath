import json
import os
import warnings

import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from datasets.APS import Dataset_APS
from datasets.Survival_kfold import Dataset_Survival
from models.ABMIL.network import DAttention
from models.APS.engine_kfold import Engine
from models.APS.network import AdaPatchSelector
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.options import aps_parse_args
from utils.scheduler import define_scheduler
from utils.util import CV_Meter, set_seed

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def main(args):
    set_seed(args.seed)
    results_dir = "results/{study}/{feature}/{model}_seed_{seed}".format(
        seed=args.seed,
        study=args.study,
        model=args.model,
        feature=args.feature,
    )
    print("[log dir] results directory: ", results_dir)
    os.makedirs(results_dir, exist_ok=True)

    mil_dataset = Dataset_Survival(pt_roots=args.pt_roots, excel_file=args.excel_file)
    args.num_classes = 4
    args.n_features = mil_dataset.n_features
    args.num_folds = mil_dataset.num_folds
    selection_path = os.path.join("datasets", "selection.json")
    with open(selection_path, "r", encoding="utf-8") as file:
        selection_dict = json.load(file)
    if args.study not in selection_dict:
        raise KeyError(f"{args.study} is not configured in json")
    args.selection = selection_dict[args.study]

    meter = CV_Meter(fold=mil_dataset.num_folds)
    if args.k_start == -1:
        args.k_start = 0
    if args.k_end == -1:
        args.k_end = mil_dataset.num_folds

    for fold in range(args.k_start, args.k_end):
        splits = mil_dataset.get_split(fold)
        mil_loaders = {}
        for split, indices in splits.items():
            if split == "train":
                mil_loaders[split] = DataLoader(
                    mil_dataset,
                    batch_size=1,
                    num_workers=4,
                    pin_memory=False,
                    sampler=SubsetRandomSampler(indices),
                )
            else:
                mil_loaders[split] = DataLoader(
                    Subset(mil_dataset, indices),
                    batch_size=1,
                    num_workers=4,
                    pin_memory=False,
                )

        aps_results_dir = os.path.join(results_dir, f"fold_{fold}", "aps")
        mil_model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        engine = Engine(args, results_dir, fold, aps_results_dir)
        engine.infer_att_score(mil_model, mil_loaders)

        aps_dataset = Dataset_APS(
            pt_roots=args.pt_roots,
            excel_file=args.excel_file,
            feature=args.feature,
            att_score_dir=aps_results_dir,
            aps_index=args.aps_index,
        )
        args.n_shallow_features = aps_dataset.n_shallow_features

        aps_loaders = {}
        for split, indices in splits.items():
            if split == "train":
                aps_loaders[split] = DataLoader(
                    aps_dataset,
                    batch_size=1,
                    num_workers=4,
                    pin_memory=False,
                    sampler=SubsetRandomSampler(indices),
                )
            else:
                aps_loaders[split] = DataLoader(
                    Subset(aps_dataset, indices),
                    batch_size=1,
                    num_workers=4,
                    pin_memory=False,
                )

        aps_model = AdaPatchSelector(in_dim=args.n_shallow_features, out_dim=1)
        print("[model] trained model: APS")
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, aps_model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)

        engine.learning(aps_model, aps_loaders, criterion, optimizer, scheduler)
        engine.infer_selection(aps_model, aps_loaders, score_only=True)
        scores = engine.evaluate_with_selection_all_loaders(mil_model, aps_loaders, args.selection)
        meter.updata(scores)

    meter.save(os.path.join(results_dir, "aps_result.csv"))


if __name__ == "__main__":
    args = aps_parse_args()
    main(args)
    print("finished!")
