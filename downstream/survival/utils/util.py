import os
import csv
import random
import numpy as np
import torch


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CV_Meter:
    def __init__(self, fold):
        self.fold = fold
        self.splits = None  # will be set on first updata
        self.header = ["folds"]
        self.rows = []

    def updata(self, results):
        '''
        results: dict
        {'val': {'C-Index': 0.0, 'epoch': 0},
         'split_1': {'C-Index': 0.0, 'epoch': 0},
         'split_2': {'C-Index': 0.0, 'epoch': 0}}
        '''
        if self.splits is None:
            self.splits = list(results.keys())
            # build header: folds, epoch_val, C-Index_val, epoch_split_1, C-Index_split_1, ...
            for split in self.splits:
                self.header.append(f"epoch_{split}")
                self.header.append(f"C-Index_{split}")
        row = [len(self.rows)]
        for split in self.splits:
            row.append(results[split]['epoch'])
            row.append(round(results[split]["C-Index"], 4))
        print(row)
        self.rows.append(row)

    def save(self, path):
        print("save evaluation resluts to", path)
        if self.fold > 1 and self.splits is not None:
            means = ["mean"]
            stds = ["std"]
            # For each split, calculate mean and std for C-Index
            for split_idx, split in enumerate(self.splits):
                cindex_col = 2 + split_idx * 2  # column index for C-Index_{split}
                cindex_values = [r[cindex_col] for r in self.rows]
                mean = round(np.mean(cindex_values), 4)
                std = round(np.std(cindex_values), 4)
                means.extend(["-", mean])
                stds.extend(["-", std])
                print({f"mean_{split}": mean, f"std_{split}": std})
            self.rows.append(means)
            self.rows.append(stds)
        with open(path, "a", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerows(self.rows)
