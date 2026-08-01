import csv
import os
import pickle
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
from utils.util import set_seed

warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


DEFAULT_SELECTIONS = [
    "0_1000",
    "0_2000",
    "0_3000",
    "0_4000",
    "500_0",
    "1000_0",
    "2000_0",
    "50_950",
    "50_1950",
    "50_2950",
    "50_3950",
    "100_900",
    "100_1900",
    "100_2900",
    "100_3900",
]


def save_summary(results_dir, all_fold_results):
    summary_pkl = os.path.join(results_dir, "selection_results.pkl")
    with open(summary_pkl, "wb") as f:
        pickle.dump(all_fold_results, f)
    print(f"Save selection results to {summary_pkl}")

    csv_path = os.path.join(results_dir, "selection_results.csv")
    selections = list(next(iter(all_fold_results.values())).keys()) if all_fold_results else []
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", *selections])
        for fold, fold_results in all_fold_results.items():
            writer.writerow([fold, *[round(fold_results[selection]["C-Index"], 4) for selection in selections]])

        if selections:
            mean_row = ["mean"]
            std_row = ["std"]
            for selection in selections:
                values = [fold_results[selection]["C-Index"] for fold_results in all_fold_results.values()]
                mean_row.append(round(sum(values) / len(values), 4))
                if len(values) == 1:
                    std_row.append(0.0)
                else:
                    mean = sum(values) / len(values)
                    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
                    std_row.append(round(std, 4))
            writer.writerow(mean_row)
            writer.writerow(std_row)
    print(f"Save selection summary to {csv_path}")


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

    if args.k_start == -1:
        args.k_start = 0
    if args.k_end == -1:
        args.k_end = mil_dataset.num_folds

    all_fold_results = {}
    for fold in range(args.k_start, args.k_end):
        splits = mil_dataset.get_split(fold)
        if "validation" not in splits:
            raise KeyError(f"Fold {fold} does not contain a validation split.")

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

        pred_score_path = os.path.join(aps_results_dir, "pred_score.pt")
        pred_att_score = torch.load(pred_score_path, weights_only=False)

        fold_results = {}
        for selection in DEFAULT_SELECTIONS:
            score, _ = engine.evaluate_with_selection_loader(
                mil_model,
                aps_loaders["validation"],
                pred_att_score,
                selection,
                split="validation",
            )
            fold_results[selection] = {"C-Index": score, "epoch": engine.best_epoch}
            print()

        fold_save_path = os.path.join(aps_results_dir, "selection_results.pkl")
        with open(fold_save_path, "wb") as f:
            pickle.dump(fold_results, f)
        print(f"Save fold selection results to {fold_save_path}")
        all_fold_results[f"fold_{fold}"] = fold_results

    save_summary(results_dir, all_fold_results)


if __name__ == "__main__":
    args = aps_parse_args()
    main(args)
    print("finished!")
