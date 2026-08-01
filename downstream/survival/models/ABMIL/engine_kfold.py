import os
import glob
import pickle
import copy
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored


import torch

class Engine(object):
    def __init__(self, args, results_dir, splits, fold=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold = fold
        self.results = {
            split: {"C-Index": 0.0, "epoch": 0}
            for split in splits.keys()
            if split != "train"
        }
        self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        os.makedirs(self.results_dir, exist_ok=True)

        self.filename_best = None
        self.best_epoch = 0
        self.epoch = 0
        self.best_ckpt = None
        self.best_outputs = None

    def learning(self, model, dataloaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            print("=> loading checkpoint '{}'".format(self.args.resume))
            ckpt = torch.load(self.args.resume, weights_only=False)
            if "results" in ckpt.keys():
                for split in self.results.keys():
                    if split in ckpt["results"].keys():
                        self.results[split] = ckpt["results"][split]
            print({k: round(v["C-Index"], 4) for k, v in self.results.items()})
            self.epoch = self.best_epoch = ckpt["epoch"] if "epoch" in ckpt.keys() else ckpt["best_epoch"]
            model.load_state_dict(ckpt["state_dict"])
            print("=> loaded checkpoint (epoch {})".format(self.epoch))
        if self.args.evaluate:
            outputs = {}
            for split in self.results.keys():
                c_index, split_outputs = self.validate(
                    dataloaders[split],
                    model,
                    criterion,
                    status=split,
                )
                outputs[split] = split_outputs
                self.results[split] = {"C-Index": c_index,
                                       "epoch": self.best_epoch}

            self.save_outputs(outputs)
            return self.results

        for epoch in range(self.epoch, self.args.num_epoch):
            print("Epoch: {}".format(epoch))
            self.epoch = epoch
            self.train(dataloaders["train"], model, criterion, optimizer)
            # evaluate
            c_index, validation_outputs = self.validate(dataloaders["validation"], model, criterion, status="validation")
            if c_index > self.results["validation"]["C-Index"]:
                outputs = {"validation": validation_outputs}
                self.results["validation"] = {"C-Index": c_index, "epoch": self.epoch}
                for split in self.results.keys():
                    if split == "validation":
                        continue
                    split_c_index, split_outputs = self.validate(
                        dataloaders[split],
                        model,
                        criterion,
                        status=split,
                    )
                    outputs[split] = split_outputs
                    self.results[split] = {"C-Index": split_c_index, "epoch": self.epoch}

                self.best_epoch = self.epoch
                self.best_ckpt = {
                    "best_epoch": self.best_epoch,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "results": self.results,
                }
                self.best_outputs = outputs

                print(f"** best epoch: {self.best_epoch}, \n best C-Index: {self.results['validation']['C-Index']}")

            for split in self.results.keys():
                print(" *** best C-Index results on {} split: {} at epoch {}".format(split, self.results[split]["C-Index"], self.best_epoch))
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")

        self.save_checkpoint(self.best_ckpt)
        self.save_outputs(self.best_outputs)
        return self.results

    def train(self, data_loader, model, criterion, optimizer):
        print("running train...")
        model.train()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch)) if self.args.tqdm else data_loader
        for batch_idx, (data_ID, data_Slide, data_WSI, data_censor, data_time, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device) if data_WSI.dtype == torch.float32 else data_WSI.to(self.device).float()
            data_Label = data_Label.to(self.device)
            data_censor = data_censor.to(self.device)

            # prediction
            logit = model(data_WSI)
            hazards = torch.sigmoid(logit)
            S = torch.cumprod(1 - hazards, dim=1)

            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_censor)

            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            # results
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_censor.item()
            all_event_times[batch_idx] = data_time.item()
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))


    def validate(self, data_loader, model, criterion, status="validation"):
        print("running {}...".format(status))
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc="{} epoch {}".format(status, self.epoch)) if self.args.tqdm else data_loader
        cases, slides, logits, risk_scores, censor, event_time, disc_label = [], [], [], [], [], [], []
        for batch_idx, (data_ID, data_Slide, data_WSI, data_censor, data_time, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device) if data_WSI.dtype == torch.float32 else data_WSI.to(self.device).float()
            data_Label = data_Label.to(self.device)
            data_censor = data_censor.to(self.device)

            with torch.no_grad():
                logit = model(data_WSI)
                hazards = torch.sigmoid(logit)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_censor)

            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_censor.item()
            all_event_times[batch_idx] = data_time.item()
            total_loss += loss.item()
            cases.append(normalize_case_id(data_ID))
            slides.append(normalize_slide_id(data_Slide))
            logits.append(logit.detach().cpu().numpy())
            risk_scores.append(float(risk.reshape(-1)[0]))
            censor.append(int(data_censor.item()))
            event_time.append(float(data_time.item()))
            disc_label.append(int(data_Label.item()))
        # calculate loss
        loss = total_loss / len(dataloader)

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        outputs = {
            "cases": cases,
            "slides": slides,
            "logits": np.concatenate(logits, axis=0),
            "risk": np.array(risk_scores),
            "censor": np.array(censor),
            "time": np.array(event_time),
            "disc_label": np.array(disc_label),
        }
        return c_index, outputs

    def evaluate_topk(self, model, loader, k_list=None, save_dir=None, mode="top", key="validation"):
        assert mode in ["top", "uniform"]
        if k_list is None:
            k_list = [50, 100, 200, 300, 500, 1000, 2000, 3000]
        k_list = list(k_list)
        if mode == "top" and -1 not in k_list:
            k_list.append(-1)

        if torch.cuda.is_available():
            model = model.cuda()
        self._load_best_model(model)

        save_dir = save_dir or self.results_dir
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f"outputs_{key}_{mode}.pkl")
        if os.path.exists(file_name):
            print(f"==== {file_name} already exists. Load as initial outputs...")
            with open(file_name, "rb") as f:
                outputs_key = pickle.load(f)
        else:
            outputs_key = {}

        scores_by_k = {}
        for k in k_list:
            if k == -1 and mode == "uniform":
                continue

            cached = outputs_key.get(k)
            if cached is not None:
                print(f"==== {mode}{k} already exists. Skip...")
                score = self._compute_c_index_from_outputs(cached)
                scores_by_k[k] = {"C-Index": float(score), "epoch": self.best_epoch}
                continue

            score, outputs = self.evaluate_topk_one_loader(model, loader, k=k, mode=mode, key=key)
            print(f"{mode}{k} {key} c_index: {score:.4f}")
            outputs_key[k] = outputs
            scores_by_k[k] = {"C-Index": float(score), "epoch": self.best_epoch}

        with open(file_name, "wb") as f:
            pickle.dump(outputs_key, f)
        print(f"Save results to {file_name}")
        return scores_by_k

    def evaluate_topk_one_loader(self, model, data_loader, k=50, mode="top", key="validation"):
        assert mode in ["top", "uniform"]
        model.eval()
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))

        dataloader = tqdm(data_loader, desc=f"{mode} {k} {key}") if self.args.tqdm else data_loader
        cases, slides, logits, risk_scores, censor, event_time, disc_label = [], [], [], [], [], [], []
        for batch_idx, (data_ID, data_Slide, data_WSI, data_censor, data_time, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device) if data_WSI.dtype == torch.float32 else data_WSI.to(self.device).float()
            data_Label = data_Label.to(self.device)
            data_censor = data_censor.to(self.device)

            with torch.no_grad():
                attn = model(data_WSI, attn_only=True)
                selected_indices = self._select_patch_indices(attn.squeeze(0), data_WSI.shape[1], k=k, mode=mode)
                selected_wsi = data_WSI[:, selected_indices]
                logit = model(selected_wsi)
                hazards = torch.sigmoid(logit)
                survival = torch.cumprod(1 - hazards, dim=1)

            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_censor.item()
            all_event_times[batch_idx] = data_time.item()

            cases.append(normalize_case_id(data_ID))
            slides.append(normalize_slide_id(data_Slide))
            logits.append(logit.detach().cpu().numpy())
            risk_scores.append(float(risk.reshape(-1)[0]))
            censor.append(int(data_censor.item()))
            event_time.append(float(data_time.item()))
            disc_label.append(int(data_Label.item()))

        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool),
            all_event_times,
            all_risk_scores,
            tied_tol=1e-08,
        )[0]
        outputs = {
            "cases": cases,
            "slides": slides,
            "logits": np.concatenate(logits, axis=0),
            "risk": np.array(risk_scores),
            "censor": np.array(censor),
            "time": np.array(event_time),
            "disc_label": np.array(disc_label),
            "selected_k": k,
            "mode": mode,
        }
        return c_index, outputs

    def _compute_c_index_from_outputs(self, outputs):
        return concordance_index_censored(
            (1 - np.asarray(outputs["censor"])).astype(bool),
            np.asarray(outputs["time"]),
            np.asarray(outputs["risk"]),
            tied_tol=1e-08,
        )[0]

    def _load_best_model(self, model):
        ckpt_path = self._find_best_ckpt()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        self.best_epoch = ckpt["best_epoch"] if "best_epoch" in ckpt else ckpt.get("epoch", 0)
        self.filename_best = ckpt_path
        print(f"=> load best mil model from {ckpt_path}")

    def _find_best_ckpt(self):
        if self.args.resume is not None:
            return self.args.resume
        candidates = glob.glob(os.path.join(self.results_dir, "*.pth.tar"))
        assert len(candidates) == 1, f"Expected 1 checkpoint in {self.results_dir}, got {candidates}"
        return candidates[0]

    def _select_patch_indices(self, attn, num_patches, k=50, mode="top"):
        if k == -1 or k >= num_patches:
            return torch.arange(num_patches, device=attn.device)

        select_num = min(k, num_patches)
        if mode == "top":
            return torch.topk(attn, k=select_num, dim=-1).indices.reshape(-1)
        if mode == "uniform":
            return torch.linspace(0, num_patches - 1, steps=select_num, device=attn.device).long()
        raise ValueError(f"Invalid mode: {mode}")

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, "model_best_{epoch}.pth.tar".format(epoch=self.best_epoch))
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def save_outputs(self, outputs):
        for split, value in outputs.items():
            file_name = os.path.join(self.results_dir, f"outputs_{split}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)


def normalize_case_id(case_id):
    case_id = case_id[0]
    if isinstance(case_id, torch.Tensor):
        return str(case_id.item())
    return str(case_id)


def normalize_slide_id(slide_id):
    slide_id = slide_id[0]
    return str(slide_id)
