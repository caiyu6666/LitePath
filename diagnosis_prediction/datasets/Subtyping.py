import os
from re import S
import pandas as pd
import socket
from tqdm import tqdm

import torch
import torch.utils.data as data


class Dataset_Subtyping(data.Dataset):
    def __init__(self, root, csv_file, feature):
        self.feature = feature
        # there are multiple roots for some specific datasets
        if "," in root:
            self.root = root.split(",")
        else:
            self.root = [root]

        # TODO: data root
        if socket.gethostname() == "jhcpu6":
            self.root = [root.replace("/ssd/", "/jhcnas1/caiyu/") for root in self.root]
        print(self.root)

        self.csv_file = csv_file
        # self.data = pd.read_csv(csv_file)
        self.data = pd.read_excel(csv_file)
        # if there is only one fold, then it is a fixed split
        if "split" in self.data.columns:
            self.split = "fixed"
            self.num_folds = 1
        else:
            self.split = "5foldcv"
            self.num_folds = 5
        # convert "label" column to discrete values
        self.data["label"] = pd.Categorical(self.data["label"])
        self.data["label"] = self.data["label"].cat.codes
        # get number of classes
        self.num_classes = len(self.data["label"].unique())
        # get the dimension of WSI features from "slide" column

        example_feature = os.path.splitext(str(self.data["slide"].values[0]).split("/")[0])[0] + ".pt"
        self.n_features = None
        for root in self.root:
            # path = os.path.join(root, self.feature, str(self.data["slide"].values[0]).split("/")[0] + ".pt")
            path = os.path.join(root, self.feature, example_feature)
            if os.path.exists(path):
                self.n_features = torch.load(path, weights_only=True).shape[-1]
                break
        if self.n_features is None:
            print(self.root)
            raise ValueError(f"No valid feature found for {example_feature}")

        self.cases = []
        for idx in range(len(self.data)):
            case = self.data.loc[idx, ['case', 'slide', 'label']].tolist()
            self.cases.append(case)
        print("[dataset] dataset from %s" % (self.csv_file))
        print("[dataset] number of cases=%d" % (len(self.cases)))
        print("[dataset] number of classes=%d" % (self.num_classes))
        print("[dataset] number of features=%d" % self.n_features)
        if self.split == "5foldcv":
            self.train = []
            self.test = []
            self.external = None
            for fold in range(5):
                split = self.data["fold{}".format(fold + 1)].values.tolist()
                train_split = [i for i, x in enumerate(split) if x == "train"]
                test_split = [i for i, x in enumerate(split) if x == "test"]
                self.train.append(train_split)
                self.test.append(test_split)
                print("[dataset] fold %d, training split: %d, test split: %d" % (fold, len(train_split), len(test_split)))
        else:
            split = self.data["split"].values.tolist()
            self.train = [i for i, x in enumerate(split) if x == "train"]
            self.val = [i for i, x in enumerate(split) if x == "val"]
            self.test = [i for i, x in enumerate(split) if x == "test"]
            print("[dataset] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))

            external_name = [s for s in self.data["split"].unique() if s not in ['train', 'val', 'test']]
            if len(external_name) == 0:
                self.external = None
                print("[dataset] no external split")
            else:
                self.external = {key: [i for i, x in enumerate(split) if x == key] for key in external_name}
                for key, value in self.external.items():
                    print("[dataset] external split {}: {}".format(key, len(value)))

    def get_fold(self, fold=0):
        if self.split == "fixed":
            assert fold == 0, "fold should be 0"
            print("[fetch *] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))
            print("[fetch *] external split:", end=" ")
            if self.external is not None:
                for key, value in self.external.items():
                    print("\n  {}: {}".format(key, len(value)), end=" ")
                print()
            else:
                print("None")
            return self.train, self.val, self.test, self.external

        elif self.split == "5foldcv":
            assert 0 <= fold <= 4, "fold should be in 0 ~ 4"
            print("[fetch *] fold %d, training split: %d, test split: %d" % (fold, len(self.train[fold]), len(self.test[fold])))
            return self.train[fold], self.test[fold]

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide, Label = case
        slide = []
        for root in self.root:
            for s in str(Slide).split("/"):
                pt_path = os.path.join(root, self.feature, os.path.splitext(s)[0] + ".pt")
                # print(pt_path)
                if os.path.exists(pt_path):
                    slide.append(torch.load(pt_path, weights_only=True))
        if len(slide) == 0:
            print(case, pt_path)
            raise ValueError("No valid slide found")

        feature = torch.cat(slide, dim=0)
        if type(feature) is not torch.Tensor:
            raise ValueError("Slide is not a tensor")
        Label = torch.tensor(Label, dtype=torch.int64)
        return ID, Slide, feature, Label

    def __len__(self):
        return len(self.cases)


class Dataset_Subtyping_Selected(data.Dataset):
    def __init__(self, csv_file, selected_feat_dir, selected_num):
        # selected_feat_path = os.path.join(selected_feat_dir, f"selected_features_select{selected_num}.pt")
        selected_feat_path = os.path.join(selected_feat_dir, f"selected_features_combine1_{selected_num}.pt")
        # selected_feat_path = os.path.join(selected_feat_dir, f"selected_features_mix2_{selected_num}.pt")
        self.selected_feat = torch.load(selected_feat_path, weights_only=True)
        print(f"=> load selected features from {selected_feat_path}")

        self.csv_file = csv_file
        self.data = pd.read_excel(csv_file)

        # convert "label" column to discrete values
        self.data["label"] = pd.Categorical(self.data["label"])
        self.data["label"] = self.data["label"].cat.codes
        # get number of classes
        self.num_classes = len(self.data["label"].unique())
        # get the dimension of WSI features from "slide" column
        example_feature = next(iter(self.selected_feat.keys()))  # get the first key
        self.n_features = self.selected_feat[example_feature].shape[-1]
        print(f"=> number of features={self.n_features}")

        self.cases = []
        for idx in range(len(self.data)):
            case = self.data.loc[idx, ['case', 'slide', 'label']].tolist()
            self.cases.append(case)
        print("[dataset] dataset from %s" % (self.csv_file))
        print("[dataset] number of cases=%d" % (len(self.cases)))
        print("[dataset] number of classes=%d" % (self.num_classes))
        print("[dataset] number of features=%d" % self.n_features)

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide, Label = case

        key = f"{ID}_{Slide}"
        feature = self.selected_feat[key]
        Label = torch.tensor(Label, dtype=torch.int64)
        return {'case_id': ID, 'slide': Slide, 'feature': feature, 'label': Label}

    def __len__(self):
        return len(self.cases)


class Dataset_Subtyping_with_Att_Score(data.Dataset):
    def __init__(self, root, csv_file, feature, att_score_dir):
        self.feature = feature
        self.att_score_dir = att_score_dir
        # there are multiple roots for some specific datasets
        if "," in root:
            self.root = root.split(",")
        else:
            self.root = [root]

        # TODO: data root
        if socket.gethostname() == "jhcpu6":
            self.root = [root.replace("/ssd/", "/jhcnas1/caiyu/") for root in self.root]
        print(self.root)

        self.pred_score = torch.load(os.path.join(self.att_score_dir, "pred_score.pt"), weights_only=True)
        print(f"=> load pred_score from {self.att_score_dir}")

        self.csv_file = csv_file
        # self.data = pd.read_csv(csv_file)
        self.data = pd.read_excel(csv_file)
        # if there is only one fold, then it is a fixed split
        if "split" in self.data.columns:
            self.split = "fixed"
            self.num_folds = 1
        else:
            self.split = "5foldcv"
            self.num_folds = 5
        # convert "label" column to discrete values
        self.data["label"] = pd.Categorical(self.data["label"])
        self.data["label"] = self.data["label"].cat.codes
        # get number of classes
        self.num_classes = len(self.data["label"].unique())
        # get the dimension of WSI features from "slide" column

        example_feature = os.path.splitext(str(self.data["slide"].values[0]).split("/")[0])[0] + ".pt"
        for root in self.root:
            # path = os.path.join(root, self.feature, str(self.data["slide"].values[0]).split("/")[0] + ".pt")
            path = os.path.join(root, self.feature, example_feature)
            if os.path.exists(path):
                self.n_features = torch.load(path, weights_only=True).shape[-1]
                break

        self.cases = []
        for idx in range(len(self.data)):
            case = self.data.loc[idx, ['case', 'slide', 'label']].tolist()
            self.cases.append(case)
        print("[dataset] dataset from %s" % (self.csv_file))
        print("[dataset] number of cases=%d" % (len(self.cases)))
        print("[dataset] number of classes=%d" % (self.num_classes))
        print("[dataset] number of features=%d" % self.n_features)
        if self.split == "5foldcv":
            self.train = []
            self.test = []
            self.external = None
            for fold in range(5):
                split = self.data["fold{}".format(fold + 1)].values.tolist()
                train_split = [i for i, x in enumerate(split) if x == "train"]
                test_split = [i for i, x in enumerate(split) if x == "test"]
                self.train.append(train_split)
                self.test.append(test_split)
                print("[dataset] fold %d, training split: %d, test split: %d" % (fold, len(train_split), len(test_split)))
        else:
            split = self.data["split"].values.tolist()
            self.train = [i for i, x in enumerate(split) if x == "train"]
            self.val = [i for i, x in enumerate(split) if x == "val"]
            self.test = [i for i, x in enumerate(split) if x == "test"]
            print("[dataset] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))

            external_name = [s for s in self.data["split"].unique() if s not in ['train', 'val', 'test']]
            if len(external_name) == 0:
                self.external = None
                print("[dataset] no external split")
            else:
                self.external = {key: [i for i, x in enumerate(split) if x == key] for key in external_name}
                for key, value in self.external.items():
                    print("[dataset] external split {}: {}".format(key, len(value)))

    def get_fold(self, fold=0):
        if self.split == "fixed":
            assert fold == 0, "fold should be 0"
            print("[fetch *] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))
            print("[fetch *] external split:", end=" ")
            if self.external is not None:
                for key, value in self.external.items():
                    print("\n  {}: {}".format(key, len(value)), end=" ")
                print()
            else:
                print("None")
            return self.train, self.val, self.test, self.external

        elif self.split == "5foldcv":
            assert 0 <= fold <= 4, "fold should be in 0 ~ 4"
            print("[fetch *] fold %d, training split: %d, test split: %d" % (fold, len(self.train[fold]), len(self.test[fold])))
            return self.train[fold], self.test[fold]

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide, Label = case
        slide = []
        for root in self.root:
            for s in str(Slide).split("/"):
                pt_path = os.path.join(root, self.feature, os.path.splitext(s)[0] + ".pt")
                # print(pt_path)
                if os.path.exists(pt_path):
                    slide.append(torch.load(pt_path, weights_only=True))
        if len(slide) == 0:
            print(case, pt_path)
            raise ValueError("No valid slide found")

        feature = torch.cat(slide, dim=0)
        if type(feature) is not torch.Tensor:
            raise ValueError("Slide is not a tensor")
        Label = torch.tensor(Label, dtype=torch.int64)

        key = f"{ID}_{Slide}"
        pred_score = self.pred_score[key]
        return ID, Slide, feature, Label, pred_score

    def __len__(self):
        return len(self.cases)
        