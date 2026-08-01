import os
import pandas as pd

import torch
import torch.utils.data as data

KNOWN_SLIDE_EXTENSIONS = {".svs", ".sdpc", ".tif", ".tiff", ".ndpi", ".mrxs", ".pt"}


def _strip_known_extension(slide_name):
    slide_name = str(slide_name)
    stem, ext = os.path.splitext(slide_name)
    if ext.lower() in KNOWN_SLIDE_EXTENSIONS:
        return stem
    return slide_name


class Dataset_Survival(data.Dataset):
    def __init__(self, pt_roots, excel_file):
        """
        Args:
            root (str): root directory of the dataset
            excel_file (str): excel file with annotations and splits
        """
        self.pt_roots = pt_roots
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        self.data = self.disc_label(self.data)
        self.fold_columns = sorted(
            [col for col in self.data.columns if str(col).startswith("Fold ")],
            key=lambda col: int(str(col).split("Fold ")[1]),
        )
        self.num_folds = len(self.fold_columns)
        if self.num_folds == 0:
            raise ValueError("No fold columns found. Expected columns like 'Fold 0'.")
        available_datasets = set(self.data["dataset"].astype(str).unique())
        missing_datasets = sorted(dataset for dataset in available_datasets if dataset not in self.pt_roots)
        if missing_datasets:
            raise ValueError(
                "Missing pt root for dataset(s): {}. Available pt_roots keys: {}".format(
                    missing_datasets, sorted(self.pt_roots.keys())
                )
            )
        label_dist = self.data['label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        # 检查slides是否有后缀，如有则去掉
        self.check_extension()
        #
        row = self.data.iloc[0]
        self.n_features = torch.load(os.path.join(self.pt_roots[row["dataset"]], str(row["slide"]).split("/")[0] + ".pt"), weights_only=True).shape[-1]
        print("[dataset] dataset from %s" % (self.excel_file))
        print("[dataset] number of cases=%d" % (len(self.data)))
        print("[dataset] number of features=%d" % self.n_features)


    def check_extension(self):
        def _rm_ext(row):
            slides = str(row["slide"]).split("/")
            new_slides = [_strip_known_extension(slide) for slide in slides]
            return "/".join(new_slides)
        self.data["slide"] = self.data.apply(_rm_ext, axis=1)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['time (months)'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['time (months)'].max() + eps
        q_bins[0] = rows['time (months)'].min() - eps
        disc_labels, q_bins = pd.cut(rows['time (months)'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'label', disc_labels)
        return rows

    def get_split(self, fold=0):
        assert 0 <= fold < self.num_folds, "fold should be in 0 ~ {}".format(self.num_folds - 1)
        splits = [str(split).strip() for split in self.data[self.fold_columns[fold]].values.tolist()]
        split_dict = {}
        if "train" in splits:
            split_dict["train"] = [i for i, x in enumerate(splits) if x == "train"]
            print("split train: {} cases".format(len(split_dict["train"])))
        if "validation" in splits:
            split_dict["validation"] = [i for i, x in enumerate(splits) if x == "validation"]
            print("split validation: {} cases".format(len(split_dict["validation"])))

        external_names = []
        for split in splits:
            if split not in ["train", "validation"] and split not in external_names:
                external_names.append(split)
        for split in external_names:
            split_dict[split] = [i for i, x in enumerate(splits) if x == split]
            print("split {}: {} cases".format(split, len(split_dict[split])))
        return split_dict

    def _load_pt_file(self, dataset, case, slides):
        pt_file = []
        # print(os.path.join(self.pt_roots[dataset], slides + ".pt"))
        if len(str(slides).split("/")) == 1:
            if os.path.exists(os.path.join(self.pt_roots[dataset], str(slides) + ".pt")):
                pt_file = [torch.load(os.path.join(self.pt_roots[dataset], str(slides) + ".pt"), weights_only=True)]
        else:
            for slide in str(slides).split("/"):
                if os.path.exists(os.path.join(self.pt_roots[dataset], str(slide) + ".pt")):
                    pt_file.append(torch.load(os.path.join(self.pt_roots[dataset], str(slide) + ".pt"), weights_only=True))
        if len(pt_file) == 0:
            raise ValueError("No slide found for case: %s" % slides)
        # print(slides)
        pt_file = torch.cat(pt_file, dim=0).to(dtype=torch.float32)
        return pt_file

    def __getitem__(self, index):
        row = self.data.iloc[index]
        dataset_ = row["dataset"]
        case_, slide_, censor_, time_, label_ = str(row["case"]), str(row["slide"]), 1 - int(row["status"]), row["time (months)"], row["label"]
        pt_file = self._load_pt_file(dataset_, case_, slide_)
        censor_ = torch.tensor(censor_, dtype=torch.int64)
        time_ = torch.tensor(time_, dtype=torch.float32)
        label_ = torch.tensor(label_, dtype=torch.int64)
        return case_, slide_, pt_file, censor_, time_, label_

    def __len__(self):
        return len(self.data)
