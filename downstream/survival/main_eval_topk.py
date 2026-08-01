import csv
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader, Subset

from datasets.Survival_kfold import Dataset_Survival
from models.ABMIL.network import DAttention
from models.ABMIL.engine_kfold import Engine
from utils.options import parse_args
from utils.util import set_seed

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.simplefilter("ignore")


def save_topk_summary(csv_path, rows):
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "mode", "k", "epoch", "c_index"])
        for row in rows:
            writer.writerow(row)
    print(f"Save topk summary to {csv_path}")


def build_eval_loaders(dataset, splits):
    loaders = {}
    for split, indices in splits.items():
        if split == "train":
            continue
        loaders[split] = DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            num_workers=4,
            pin_memory=False,
        )
    return loaders


def main(args):
    set_seed(args.seed)
    if args.resume is not None:
        resume_paths = args.resume.split(";")
        results_dir = os.path.dirname(os.path.dirname(resume_paths[0]))
    else:
        resume_paths = None
        results_dir = "results/{study}/{feature}/{model}_seed_{seed}".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
        )

    print("[log dir] results directory: ", results_dir)

    dataset = Dataset_Survival(pt_roots=args.pt_roots, excel_file=args.excel_file)
    args.num_classes = 4
    num_folds = dataset.num_folds
    if args.k_start == -1:
        args.k_start = 0
    if args.k_end == -1:
        args.k_end = num_folds

    if args.model != "ABMIL":
        raise NotImplementedError("model [{}] is not implemented".format(args.model))

    k_list = [50, 100, 200, 300, 500, 1000, 2000, 3000]
    summary_rows = []
    for fold in range(args.k_start, args.k_end):
        splits = dataset.get_split(fold)
        eval_loaders = build_eval_loaders(dataset, splits)
        fold_results_dir = os.path.join(results_dir, f"fold_{fold}", "uniformk")
        os.makedirs(fold_results_dir, exist_ok=True)
        fold_summary_rows = []

        args.resume = None if resume_paths is None else resume_paths[fold]
        model = DAttention(
            n_classes=args.num_classes,
            dropout=0.25,
            act="relu",
            n_features=dataset.n_features,
        )
        engine = Engine(args, results_dir, splits, fold)

        for split, loader in eval_loaders.items():
            for mode in ["uniform", "top"]:
                scores = engine.evaluate_topk(
                    model,
                    loader,
                    k_list=k_list,
                    save_dir=fold_results_dir,
                    mode=mode,
                    key=split,
                )
                for k, result in scores.items():
                    row = [fold, split, mode, k, result["epoch"], round(result["C-Index"], 6)]
                    fold_summary_rows.append(row)
                    summary_rows.append(row)

        save_topk_summary(os.path.join(fold_results_dir, "cindex_summary.csv"), fold_summary_rows)

    save_topk_summary(os.path.join(results_dir, "cindex_summary_all_folds.csv"), summary_rows)


if __name__ == "__main__":
    start_time = time.strftime("[%Y-%m-%d]-[%H-%M-%S]")
    print(f"======================================= Start Evaluation at {start_time} =======================================")

    args = parse_args()
    main(args)
    print("finished!")
