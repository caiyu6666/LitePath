import copy
import glob
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm


class Engine(object):
    def __init__(self, args, results_dir, fold, aps_results_dir):
        self.args = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        self.aps_results_dir = aps_results_dir
        os.makedirs(self.aps_results_dir, exist_ok=True)

        self.filename_best = None
        self.mil_filename_best = None
        self.best_epoch = 0
        self.best_ckpt = None
        self.early_stop = 0
        self.val_scores = None
        self.att_score = {}

    def learning(self, model, loaders, criterion, optimizer, scheduler):
        print(">>>")
        print(">>>")
        print("**************************** start learning APS *****************************")
        self.filename_best = os.path.join(self.aps_results_dir, "aps_model_best.pth.tar")
        if os.path.exists(self.filename_best):
            print(f"==== {self.filename_best} already exists. Skip...")
            return

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(self.best_epoch, self.args.num_epoch):
            print(f"--------------------------------Epoch {epoch} / {self.args.num_epoch}--------------------------------")
            self.epoch = epoch
            train_loader = loaders["train"]
            val_loader = loaders["validation"]
            train_loss = self.train(train_loader, model, criterion, optimizer)
            val_scores = self.validate(val_loader, model)
            print(f"Train Loss: {train_loss}")
            print("Val Scores:")
            self.print_scores(val_scores)

            is_best = (val_scores["softmax_sum"] > self.val_scores["softmax_sum"]) if self.val_scores is not None else True
            if is_best:
                self.val_scores = val_scores
                self.best_epoch = self.epoch
                self.best_ckpt = {
                    "best_epoch": self.best_epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "val_scores": self.val_scores,
                }
                print(f"best epoch: {self.best_epoch}, best val scores: {self.val_scores}")

            print(f"** best epoch: {self.best_epoch}, best val scores: {self.val_scores}")
            if scheduler is not None:
                scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")

            if is_best:
                self.early_stop = 0
            else:
                self.early_stop += 1
            if self.early_stop >= 20:
                print("Early stopping")
                break
        self.save_checkpoint(self.best_ckpt)

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for idx, data_item in enumerate(data_loader):
            if idx % 50 == 0:
                print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
            data = data_item["shallow_feature"].squeeze(0).to(self.device)
            att_score = data_item["att_score"].squeeze().to(self.device)
            outputs = model(data).squeeze()
            loss = criterion(outputs, att_score)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / len(data_loader)

    def validate(self, data_loader, model):
        metrics_dict = {"top10_recall": [], "top250_recall": [], "top500_recall": [], "softmax_sum": [], "reference_sum": []}
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, data_item in enumerate(data_loader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
                data = data_item["shallow_feature"].squeeze(0).to(self.device)
                att_score = data_item["att_score"].squeeze().to(self.device)
                outputs = model(data).squeeze()
                metrics = self.metrics(outputs, att_score)
                for key, value in metrics.items():
                    metrics_dict[key].append(value)
        return {k: np.mean(v) for k, v in metrics_dict.items()}

    def metrics(self, pred_score, att_score):
        assert pred_score.shape[0] == att_score.shape[0]
        num_patches = pred_score.shape[0]
        pred_topk = torch.topk(pred_score, k=min(500, num_patches)).indices
        label_top10 = torch.topk(att_score, k=min(10, num_patches)).indices
        label_top250 = torch.topk(att_score, k=min(250, num_patches)).indices
        label_top500 = torch.topk(att_score, k=min(500, num_patches)).indices

        att_score_softmax = F.softmax(att_score, dim=0)
        return {
            "top10_recall": recall(label_top10, pred_topk),
            "top250_recall": recall(label_top250, pred_topk),
            "top500_recall": recall(label_top500, pred_topk),
            "softmax_sum": att_score_softmax[pred_topk].sum().item(),
            "reference_sum": att_score_softmax[label_top500].sum().item(),
        }

    def infer_att_score(self, mil_model, mil_loaders):
        print(">>>")
        print(">>>")
        print("************************* Infer att_score of the best mil model *************************")
        save_path = os.path.join(self.aps_results_dir, "att_score.pt")
        if os.path.exists(save_path):
            print(f"==== att_score already exists in {save_path}. Skip...")
            return

        if torch.cuda.is_available():
            mil_model = mil_model.cuda()
        self.mil_filename_best = self.find_best_mil_ckpt()
        state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=False)["state_dict"]
        mil_model.load_state_dict(state_dict)
        print(f"=> load best mil model from {self.mil_filename_best}")

        for split, loader in mil_loaders.items():
            self.infer_att_score_one_loader(mil_model, loader, split=split)
        torch.save(self.att_score, save_path)
        print(f"=> save att_score to {save_path}")
        self.att_score = None

    def infer_att_score_one_loader(self, mil_model, mil_dataloader, split="train"):
        print(f"=> infer {split} loader attn score ... ")
        mil_model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, (data_id, data_slide, data_wsi, _, _, _) in enumerate(mil_dataloader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(mil_dataloader)} time: {time.time() - t0:.2f}s")
                data_wsi = data_wsi.to(self.device).float()
                _, attn = mil_model(data_wsi, return_attn=True)
                key = f"{normalize_case_id(data_id)}_{str(data_slide[0])}"
                self.att_score[key] = attn.cpu()

    def infer_selection(self, model, loaders, score_only=False):
        print(">>>")
        print(">>>")
        print("************************* Infer APS to get selected features *************************")
        pred_score_path = os.path.join(self.aps_results_dir, "pred_score.pt")
        selected_features_path = os.path.join(self.aps_results_dir, f"selected_features.pt")

        if os.path.exists(selected_features_path) and not score_only:
            print(f"==== {selected_features_path} already exists. Skip...")
            return

        if os.path.exists(pred_score_path):
            pred_att_score = torch.load(pred_score_path, weights_only=False)
            print(f"=> load pred_att_score from {pred_score_path}")
        else:
            pred_att_score = {}
            if torch.cuda.is_available():
                model = model.cuda()
            model.load_state_dict(torch.load(self.filename_best, map_location=self.device, weights_only=False)["state_dict"])
            print(f"=> load best aps model from {self.filename_best}")
            for split, loader in loaders.items():
                pred_att_score.update(self.infer_pred_score_one_loader(model, loader, split=split))
            torch.save(pred_att_score, pred_score_path)
            print(f"=> save pred_att_score to {pred_score_path}")

        if score_only:
            return

        selected_features = {}
        for split, loader in loaders.items():
            if split == "train":
                continue
            selected_features.update(self.get_selected_features_one_loader(pred_att_score, loader, split=split))
        torch.save(selected_features, selected_features_path)
        print(f"=> save selected_features to {selected_features_path}")

    def infer_pred_score_one_loader(self, model, data_loader, split="test"):
        print(f"=> infer {split} loader pred_score ... ")
        pred_att_score = {}
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, data_item in enumerate(data_loader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
                data = data_item["shallow_feature"].squeeze(0).to(self.device)
                outputs = model(data)
                key = f"{normalize_case_id(data_item['case_id'])}_{str(data_item['slide'][0])}"
                pred_att_score[key] = outputs.cpu()
        return pred_att_score

    def get_selected_features_one_loader(self, pred_att_score, data_loader, split="test"):
        print(f"=> get {split} loader selected features ... ")
        selected_features = {}
        attention_num, uniform_num = self.parse_selection(self.args.selection)
        t0 = time.time()
        with torch.no_grad():
            for idx, data_item in enumerate(data_loader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
                feature = data_item["feature"][0]
                key = f"{normalize_case_id(data_item['case_id'])}_{str(data_item['slide'][0])}"
                pred_score = pred_att_score[key].squeeze()
                assert feature.shape[0] == pred_score.shape[0]
                selected_features[key] = feature[self.select_indices(pred_score, feature.shape[0], attention_num, uniform_num)]
        return selected_features

    def evaluate_with_selection_all_loaders(self, mil_model, aps_loaders, selection):
        print("Selection: ", selection)
        if torch.cuda.is_available():
            mil_model = mil_model.cuda()

        self.mil_filename_best = self.find_best_mil_ckpt()
        state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=False)["state_dict"]
        mil_model.load_state_dict(state_dict)
        print(f"=> load best mil model from {self.mil_filename_best}")

        pred_score_path = os.path.join(self.aps_results_dir, "pred_score.pt")
        pred_att_score = torch.load(pred_score_path, weights_only=False)
        outputs = {}
        results = {}
        for split, loader in aps_loaders.items():
            if split == "train":
                continue
            scores, split_outputs = self.evaluate_with_selection_one_loader(
                loader, mil_model, pred_att_score, selection, split=split
            )
            results[split] = {"C-Index": scores, "epoch": self.best_epoch}
            outputs[split] = split_outputs
        self.save_outputs(outputs)
        return results

    def evaluate_with_selection_loader(self, mil_model, eval_loader, pred_att_score, selection, split="validation"):
        print("Selection: ", selection)
        if torch.cuda.is_available():
            mil_model = mil_model.cuda()

        if self.mil_filename_best is None:
            self.mil_filename_best = self.find_best_mil_ckpt()
            state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=False)["state_dict"]
            mil_model.load_state_dict(state_dict)
            print(f"=> load best mil model from {self.mil_filename_best}")
        return self.evaluate_with_selection_one_loader(eval_loader, mil_model, pred_att_score, selection, split=split)

    def evaluate_with_selection_one_loader(self, eval_loader, mil_model, pred_att_score, selection, split="test"):
        mil_model.eval()
        total_loss = 0.0
        all_risk_scores, all_censorships, all_event_times = [], [], []
        cases, slides, logits, risk_scores, censor, event_time, disc_label = [], [], [], [], [], [], []
        attention_num, uniform_num = self.parse_selection(selection)
        print(f"=> evaluate with selection: {attention_num} APS and {uniform_num} uniform")

        t0 = time.time()
        with torch.no_grad():
            for idx, data_item in enumerate(eval_loader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(eval_loader)} time: {time.time() - t0:.2f}s")
                feature = data_item["feature"][0]
                case_id = normalize_case_id(data_item["case_id"])
                slide = str(data_item["slide"][0])
                key = f"{case_id}_{slide}"
                pred_score = pred_att_score[key].squeeze()
                selected_indices = self.select_indices(pred_score, feature.shape[0], attention_num, uniform_num)
                selected_feature = feature[selected_indices].unsqueeze(0).to(self.device).float()
                data_censor = data_item["censor"].to(self.device)
                data_time = data_item["time"].to(self.device)
                data_label = data_item["label"].to(self.device)

                logit = mil_model(selected_feature)
                hazards = torch.sigmoid(logit)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

                all_risk_scores.extend(risk.tolist())
                all_censorships.extend(data_censor.detach().cpu().numpy().tolist())
                all_event_times.extend(data_time.detach().cpu().numpy().tolist())

                cases.append(case_id)
                slides.append(slide)
                logits.append(logit.detach().cpu().numpy())
                risk_scores.extend(risk.tolist())
                censor.extend(data_censor.detach().cpu().numpy().astype(int).tolist())
                event_time.extend(data_time.detach().cpu().numpy().astype(float).tolist())
                disc_label.extend(data_label.detach().cpu().numpy().astype(int).tolist())

        c_index = concordance_index_censored(
            (1 - np.asarray(all_censorships)).astype(bool),
            np.asarray(all_event_times),
            np.asarray(all_risk_scores),
            tied_tol=1e-08,
        )[0]
        print(f"=> {split} c_index: {c_index:.4f}")
        outputs = {
            "cases": cases,
            "slides": slides,
            "logits": np.concatenate(logits, axis=0),
            "risk": np.asarray(risk_scores),
            "censor": np.asarray(censor),
            "time": np.asarray(event_time),
            "disc_label": np.asarray(disc_label),
        }
        return c_index, outputs

    def parse_selection(self, selection):
        attention_num, uniform_num = selection.split("_")
        return int(attention_num), int(uniform_num)

    def select_indices(self, pred_score, num_patches, attention_num, uniform_num):
        all_indices = torch.arange(num_patches)
        if attention_num + uniform_num >= num_patches:
            return all_indices
        if uniform_num <= 0:
            return torch.topk(pred_score, k=attention_num).indices
        uniform_indices = torch.linspace(0, num_patches - 1, steps=uniform_num).int()
        remain_indices = torch.tensor([i for i in all_indices.tolist() if i not in uniform_indices.tolist()])
        if attention_num <= 0:
            return uniform_indices
        remain_scores = pred_score[remain_indices]
        topk_idx_in_remain = torch.topk(remain_scores, k=attention_num).indices
        topk_indices = remain_indices[topk_idx_in_remain]
        return torch.cat([uniform_indices, topk_indices])

    def save_checkpoint(self, state):
        if state is None:
            return
        torch.save(state, self.filename_best)
        print("save best model {filename}".format(filename=self.filename_best))

    def save_outputs(self, outputs):
        for key, value in outputs.items():
            file_name = os.path.join(self.aps_results_dir, f"outputs_{key}_best.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)
            print(f"Save outputs to {file_name}")

    def print_scores(self, scores):
        for key, value in scores.items():
            print(f"    {key}: {value:.4f}")
        print("---")

    def find_best_mil_ckpt(self):
        mil_filename_best = glob.glob(os.path.join(self.results_dir, "*.pth.tar"))
        assert len(mil_filename_best) == 1, f"Expected 1 MIL checkpoint in {self.results_dir}, got {mil_filename_best}"
        return mil_filename_best[0]


def normalize_case_id(case_id):
    if isinstance(case_id, (list, tuple)):
        case_id = case_id[0]
    if isinstance(case_id, torch.Tensor):
        return str(case_id.item())
    return str(case_id)


def recall(topk_true, pred_topk):
    num_recalled = len(set(topk_true.tolist()) & set(pred_topk.tolist()))
    return num_recalled / len(topk_true)
