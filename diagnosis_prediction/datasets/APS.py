import os
from re import S
import pandas as pd
import socket
from tqdm import tqdm

import torch
import torch.utils.data as data


class Dataset_APS(data.Dataset):
    def __init__(self, root, csv_file, feature, att_score_dir, aps_index=0):
        self.att_score_dir = att_score_dir
        self.feature = feature
        self.shallow_feature = f"{feature}-block{aps_index}"

        self.attn_scores = torch.load(os.path.join(self.att_score_dir, "att_score.pt"), weights_only=True)
        print("=> load attn scores from ", os.path.join(self.att_score_dir, "att_score.pt"))
        # there are multiple roots for some specific datasets
        if "," in root:
            self.root = root.split(",")
        else:
            self.root = [root]

        # TODO: data root
        if socket.gethostname() == "jhcpu6":
            self.root = [root.replace("/ssd", "/jhcnas1/caiyu") for root in self.root]

        self.csv_file = csv_file
        self.data = pd.read_excel(csv_file)

        example_feature = os.path.splitext(str(self.data["slide"].values[0]).split("/")[0])[0] + ".pt"
        for root in self.root:
            path_shallow = os.path.join(root, self.shallow_feature, example_feature)
            if os.path.exists(path_shallow):
                self.n_shallow_features = torch.load(path_shallow, weights_only=True).shape[-1]
                break

        self.cases = []
        for idx in range(len(self.data)):
            case = self.data.loc[idx, ['case', 'slide']].tolist()
            self.cases.append(case)
        print("[dataset] number of shallow features=%d" % self.n_shallow_features)

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide = case
        key = f"{ID}_{Slide}"
        att_score = self.attn_scores[key]

        slide = []
        shallow_slide = []
        for root in self.root:
            for s in str(Slide).split("/"):
                if os.path.exists(os.path.join(root, self.shallow_feature, os.path.splitext(s)[0] + ".pt")):
                    shallow_slide.append(torch.load(os.path.join(root, self.shallow_feature, os.path.splitext(s)[0] + ".pt"), weights_only=True))
                    slide.append(torch.load(os.path.join(root, self.feature, os.path.splitext(s)[0] + ".pt"), weights_only=True))

        if len(shallow_slide) == 0:
            print(case)
            raise ValueError("No valid slide found")

        shallow_feature = torch.cat(shallow_slide, dim=0)
        feature = torch.cat(slide, dim=0) if len(slide) > 0 else None
        if type(shallow_feature) is not torch.Tensor:
            raise ValueError("Slide is not a tensor")

        assert shallow_feature.shape[0] == att_score.shape[1], f"{key}: shallow_feature and att_score: {shallow_feature.shape}, {att_score.shape}"

        return {'case_id': ID, 'slide': Slide, 'shallow_feature': shallow_feature, 'feature': feature, 'att_score': att_score}

    def __len__(self):
        return len(self.cases)
