import os

import pandas as pd
import torch
import torch.utils.data as data

from datasets.Survival_kfold import _strip_known_extension


class Dataset_APS(data.Dataset):
    def __init__(self, pt_roots, excel_file, feature, att_score_dir, aps_index=0):
        self.pt_roots = pt_roots
        self.excel_file = excel_file
        self.feature = feature
        self.shallow_feature = f"{feature}-block{aps_index}"
        self.att_score_dir = att_score_dir

        self.attn_scores = torch.load(os.path.join(self.att_score_dir, "att_score.pt"), weights_only=True)
        print("=> load attn scores from", os.path.join(self.att_score_dir, "att_score.pt"))

        self.data = pd.read_excel(excel_file)
        self.data = self.disc_label(self.data)
        self.check_extension()

        self.fold_columns = sorted(
            [col for col in self.data.columns if str(col).startswith("Fold ")],
            key=lambda col: int(str(col).split("Fold ")[1]),
        )
        self.num_folds = len(self.fold_columns)
        if self.num_folds == 0:
            raise ValueError("No fold columns found. Expected columns like 'Fold 0'.")

        self.n_shallow_features = self._infer_shallow_feature_dim()
        self.cases = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            self.cases.append(
                {
                    "dataset": str(row["dataset"]).strip(),
                    "case": str(row["case"]),
                    "slide": str(row["slide"]),
                    "status": int(row["status"]),
                    "time": float(row["time (months)"]),
                    "label": int(row["label"]),
                }
            )
        print("[dataset] number of shallow features=%d" % self.n_shallow_features)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows["status"] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df["time (months)"], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows["time (months)"].max() + eps
        q_bins[0] = rows["time (months)"].min() - eps
        disc_labels, q_bins = pd.cut(
            rows["time (months)"],
            bins=q_bins,
            retbins=True,
            labels=False,
            right=False,
            include_lowest=True,
        )
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), "label", disc_labels)
        return rows

    def check_extension(self):
        def _rm_ext(row):
            slides = str(row["slide"]).split("/")
            new_slides = [_strip_known_extension(slide) for slide in slides]
            return "/".join(new_slides)

        self.data["slide"] = self.data.apply(_rm_ext, axis=1)

    def _infer_shallow_feature_dim(self):
        for _, row in self.data.iterrows():
            dataset = str(row["dataset"]).strip()
            for slide_name in str(row["slide"]).split("/"):
                pt_path = self._resolve_pt_path(dataset, slide_name, shallow=True)
                if pt_path is not None:
                    return torch.load(pt_path, weights_only=True).shape[-1]
        raise ValueError("No valid shallow feature found to infer feature dimension")

    def _candidate_slide_names(self, slide_name):
        slide_name = str(slide_name).strip()
        candidate_names = [slide_name]
        slide_stem = os.path.splitext(slide_name)[0]
        if slide_stem != slide_name:
            candidate_names.append(slide_stem)
        return candidate_names

    def _get_feature_root(self, dataset, shallow=False):
        base_root = self.pt_roots[dataset]
        if not shallow:
            return base_root
        root_norm = os.path.normpath(base_root)
        parent_dir = os.path.dirname(root_norm)
        return os.path.join(parent_dir, self.shallow_feature)

    def _resolve_pt_path(self, dataset, slide_name, shallow=False):
        feature_root = self._get_feature_root(dataset, shallow=shallow)
        for candidate_name in self._candidate_slide_names(slide_name):
            pt_path = os.path.join(feature_root, candidate_name + ".pt")
            if os.path.exists(pt_path):
                return pt_path
        return None

    def __getitem__(self, index):
        case = self.cases[index]
        case_id = case["case"]
        slide = case["slide"]
        key = f"{case_id}_{slide}"
        att_score = self.attn_scores[key]

        shallow_slide = []
        full_slide = []
        for slide_name in slide.split("/"):
            shallow_path = self._resolve_pt_path(case["dataset"], slide_name, shallow=True)
            full_path = self._resolve_pt_path(case["dataset"], slide_name, shallow=False)
            if shallow_path is not None:
                shallow_slide.append(torch.load(shallow_path, weights_only=True))
            if full_path is not None:
                full_slide.append(torch.load(full_path, weights_only=True))

        if len(shallow_slide) == 0:
            raise ValueError("No valid shallow slide found for case {}".format(case_id))
        if len(full_slide) == 0:
            raise ValueError("No valid full slide found for case {}".format(case_id))

        shallow_feature = torch.cat(shallow_slide, dim=0).float()
        feature = torch.cat(full_slide, dim=0).float()
        assert shallow_feature.shape[0] == att_score.shape[1], (
            f"{key}: shallow_feature and att_score shape mismatch: {shallow_feature.shape}, {att_score.shape}"
        )

        censor = torch.tensor(1 - case["status"], dtype=torch.int64)
        time_ = torch.tensor(case["time"], dtype=torch.float32)
        label = torch.tensor(case["label"], dtype=torch.int64)
        return {
            "case_id": case_id,
            "slide": slide,
            "shallow_feature": shallow_feature,
            "feature": feature,
            "att_score": att_score,
            "censor": censor,
            "time": time_,
            "label": label,
        }

    def __len__(self):
        return len(self.cases)
