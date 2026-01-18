import os
import numpy as np
from tqdm import tqdm
import pickle
import glob
import time
import copy

from tensorboardX import SummaryWriter

from typing import Dict
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.bootstrapping import BootStrapper
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC

import torch
import torch.nn.functional as F


class Engine(object):
    def __init__(self, args, results_dir, fold, att_score_dir=None):
        self.args = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.num_folds > 1:
            self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        else:
            self.results_dir = results_dir
        
        self.val_scores = None
        self.test_scores = None if self.args.num_folds > 1 else dict()
        self.filename_best = None
        self.best_epoch = 0
        self.early_stop = 0

        self.best_ckpt = None
        self.best_outputs = None

        self.att_score_dir = att_score_dir
        self.att_score = {}


    def learning(self, model, loaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        # if self.args.resume is not None:
        #     if os.path.isfile(self.args.resume):
        #         print("=> loading checkpoint '{}'".format(self.args.resume))
        #         checkpoint = torch.load(self.args.resume)
        #         self.val_scores = checkpoint["val_scores"]
        #         self.best_epoch = checkpoint["best_epoch"]
        #         if "test_score" in checkpoint:
        #             self.test_scores = checkpoint["test_scores"]
        #         model.load_state_dict(checkpoint["state_dict"])
        #         print("=> loaded checkpoint (val score: {})".format(checkpoint["val_score"]["Macro_AUC"]))
        #         if self.test_scores is not None:
        #             print("=> loaded checkpoint (test score: {})".format(checkpoint["test_score"]["Macro_AUC"]))
        #     else:
        #         print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.find_best_ckpt()
            ckpt = torch.load(self.filename_best, map_location=self.device)
            self.best_epoch = ckpt["best_epoch"]
            self.epoch = self.best_epoch
            model.load_state_dict(ckpt["state_dict"])
            print(f"=> load best model from {self.filename_best}")

            val_loader, test_loader = loaders[1:3]
            external_loaders = loaders[3]

            self.val_scores = self.validate(val_loader, model, criterion, status="val")
            self.test_scores, test_outputs = self.validate(test_loader, model, criterion, status="test")
            self.outputs = {'test': test_outputs}
            if external_loaders is not None:
                self.external_scores, self.external_outputs = {}, {}
                for key, one_external_loader in external_loaders.items():
                    self.external_scores[key], self.external_outputs[key] = self.validate(one_external_loader, model, criterion, status=key)
                    self.outputs.update(self.external_outputs)
            else:
                self.external_scores = None

            self.save_outputs(self.outputs)

            return self.val_scores, self.test_scores, self.best_epoch, self.external_scores

        for epoch in range(self.best_epoch, self.args.num_epoch):
            self.epoch = epoch
            if self.args.num_folds > 1:
                train_loader, val_loader = loaders
            else:
                train_loader, val_loader, test_loader, external_loaders = loaders
            # train
            train_scores = self.train(train_loader, model, criterion, optimizer)
            # evaluate
            val_scores = self.validate(val_loader, model, criterion, status="val")
            is_best = (val_scores["Macro_AUC"] > self.val_scores["Macro_AUC"]) if self.val_scores is not None else True
            if self.args.num_folds > 1:
                if is_best:
                    self.val_scores = val_scores
                    self.best_epoch = self.epoch
                    self.save_checkpoint(
                        {
                            "best_epoch": self.best_epoch,
                            "state_dict": model.state_dict(),
                            "val_scores": self.val_scores,
                        }
                    )
            else:
                if is_best:
                    # external_scores = {key: self.validate(one_external_loader, model, criterion, status=key) \
                    #     for key, one_external_loader in external_loaders.items()} if external_loaders else None
                    test_scores, test_outputs = self.validate(test_loader, model, criterion, status="test")
                    outputs = {'test': test_outputs}
                    if external_loaders is not None:
                        external_scores, external_outputs = {}, {}
                        for key, one_external_loader in external_loaders.items():
                            external_scores[key], external_outputs[key] = self.validate(one_external_loader, model, criterion, status=key)
                        outputs.update(external_outputs)
                    else:
                        external_scores = None
                    results = {"internal": test_scores, "external": external_scores}
                    self.val_scores = val_scores
                    self.test_scores = test_scores
                    self.external_scores = external_scores
                    self.best_epoch = self.epoch

                    self.best_ckpt = {
                            "best_epoch": self.best_epoch,
                            # "state_dict": model.state_dict(),
                            "state_dict": copy.deepcopy(model.state_dict()),
                            "val_scores": self.val_scores,
                            # "test_scores": self.test_scores,
                            'test_scores': results,
                        }
                    self.best_outputs = outputs
                    print("** update best checkpoint at epoch {}".format(self.best_epoch))
                    # self.save_checkpoint(
                    #     {
                    #         "best_epoch": self.best_epoch,
                    #         "state_dict": model.state_dict(),
                    #         "val_scores": self.val_scores,
                    #         # "test_scores": self.test_scores,
                    #         'test_scores': results,
                    #     }
                    # )
                    # self.save_outputs(outputs)
            # print(" *** best model {}".format(self.filename_best))
            print(f"** best epoch: {self.best_epoch}, \n best val scores: {self.val_scores}, \n best test scores: {self.test_scores}")
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")
            if is_best:
                self.early_stop = 0
            else:
                self.early_stop += 1
            if self.early_stop >= 10:
                print("Early stopping")
                break
        
        self.save_outputs(self.best_outputs)
        self.save_checkpoint(self.best_ckpt)
        if self.args.num_folds > 1:
            return self.val_scores, self.best_epoch
        else:
            return self.val_scores, self.test_scores, self.best_epoch, self.external_scores

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------train epoch {}-------------------------------".format(self.epoch))

        for batch_idx, (data_ID, data_Slide, data_WSI, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            # data_Label = data_Label.long().to(self.device)
            logit = model(data_WSI)
            loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics
        scores, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        return scores

    def validate(self, data_loader, model, criterion, status="val"):
        model.eval()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="{} epoch {}".format(status, self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------{} epoch {}-------------------------------".format(status, self.epoch))

        cases, slides, logits, labels, attns = [], [], [], [], []
        for batch_idx, (data_ID, data_Slide, data_WSI, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            one_hot_label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            with torch.no_grad():
                logit, attn = model(data_WSI, return_attn=True)
                loss = criterion(logit.view(1, -1), one_hot_label)
            # results
            all_labels = np.row_stack((all_labels, one_hot_label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()

            if status not in ['train', 'val']:
                cases.append(normalize_case_id(data_ID))
                slides.append(data_Slide[0])
                logits.append(logit.cpu().numpy())
                labels.append(data_Label.item())
                attns.append(attn.cpu().numpy())

        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics
        if status == "val":
            scores, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
            return scores
        else:
            # _, scores = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
            scores, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))

            logits = np.concatenate(logits, axis=0)
            labels = np.array(labels)
            # attns = np.concatenate(attns, axis=0)
            outputs = {'cases': cases, 'slides': slides, 'logits': logits, 'labels': labels, 'attns': attns}
            # # save outputs to pkl file
            # with open(os.path.join(self.results_dir, "outputs.pkl"), "wb") as f:
            #     pickle.dump(outputs, f)
            return scores, outputs

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        if self.test_scores is not None:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{test_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    # test_score=self.test_scores["Macro_AUC_mean"],
                    test_score=self.test_scores["Macro_AUC"],
                    epoch=self.best_epoch,
                ),
            )
        else:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    epoch=self.best_epoch,
                ),
            )
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def save_outputs(self, outputs):
        for key, value in outputs.items():
            file_name = os.path.join(self.results_dir, f"outputs_{key}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)

    def metrics(self, logits, labels, bootstrap=False):
        general_meter = self.meter(num_classes=self.args.num_classes, bootstrap=False)
        # general results
        general_results = general_meter(logits, labels)
        print("General Results:")
        print(
            "Macro AUC:    {:.4f},   Macro ACC:    {:.4f},   Macro F1:    {:.4f}".format(
                general_results["Macro_AUC"],
                general_results["Macro_ACC"],
                general_results["Macro_F1"],
            )
        )
        print(
            "Weighted AUC: {:.4f},   Weighted ACC: {:.4f},   Weighted F1: {:.4f}".format(
                general_results["Weighted_AUC"],
                general_results["Weighted_ACC"],
                general_results["Weighted_F1"],
            )
        )
        
        if bootstrap:
        # bootstrapped results
            bootstrapped_meter = self.meter(num_classes=self.args.num_classes, bootstrap=True)
            bootstrapped_results = bootstrapped_meter(logits, labels)
            print("Bootstrapped Results:")
            print(
                "Macro AUC:    {:.4f}±{:.4f},   Macro ACC:    {:.4f}±{:.4f},   Macro F1:    {:.4f}±{:.4f}".format(
                    bootstrapped_results["Macro_AUC_mean"],
                    bootstrapped_results["Macro_AUC_std"],
                    bootstrapped_results["Macro_ACC_mean"],
                    bootstrapped_results["Macro_ACC_std"],
                    bootstrapped_results["Macro_F1_mean"],
                    bootstrapped_results["Macro_F1_std"],
                )
            )
            print(
                "Weighted AUC: {:.4f}±{:.4f},   Weighted ACC: {:.4f}±{:.4f},   Weighted F1: {:.4f}±{:.4f}".format(
                    bootstrapped_results["Weighted_AUC_mean"],
                    bootstrapped_results["Weighted_AUC_std"],
                    bootstrapped_results["Weighted_ACC_mean"],
                    bootstrapped_results["Weighted_ACC_std"],
                    bootstrapped_results["Weighted_F1_mean"],
                    bootstrapped_results["Weighted_F1_std"],
                )
            )
        else:
            bootstrapped_results = None
        return general_results, bootstrapped_results

    def meter(self, num_classes, bootstrap=False):
        metrics: Dict[str, Metric] = {
            "Macro_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_F1": F1Score(num_classes=int(num_classes), average="macro", task="multiclass").to(self.device),
            "Macro_AUC": AUROC(num_classes=num_classes, average="macro", task="multiclass").to(self.device),
            "Weighted_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_F1": F1Score(num_classes=int(num_classes), average="weighted", task="multiclass").to(self.device),
            "Weighted_AUC": AUROC(num_classes=num_classes, average="weighted", task="multiclass").to(self.device),
        }
        # boot strap wrap
        if bootstrap:
            for k, m in metrics.items():
                # print("wrapping:", k)
                metrics[k] = BootStrapper(m, num_bootstraps=1000, sampling_strategy="multinomial").to(self.device)
        metrics = MetricCollection(metrics)
        return metrics

    def infer_att_score(self, model, loaders):
        if torch.cuda.is_available():
            model = model.cuda()
        self.find_best_ckpt()
        state_dict = torch.load(self.filename_best, map_location=self.device, weights_only=True)['state_dict']
        model.load_state_dict(state_dict)
        print(f"=> load best model from {self.filename_best}")

        train_loader, val_loader, test_loader, external_loader = loaders
        self.infer_one_loader_score(model, train_loader, key='train')
        self.infer_one_loader_score(model, val_loader, key='val')
        self.infer_one_loader_score(model, test_loader, key='test')
        if external_loader is not None:
            for key, loader in external_loader.items():
                self.infer_one_loader_score(model, loader, key=key)
        torch.save(self.att_score, os.path.join(self.att_score_dir, f"{self.args.study}.pt"))

    def infer_one_loader_score(self, model, dataloader, key='train'):
        print(f"=> infer {key} loader attn score ... ")
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, (data_ID, data_Slide, data_WSI, data_Label) in enumerate(dataloader):
                if idx % 20 == 0:
                    print(f"=> infer {key} loader attn score ... {idx}/{len(dataloader)} time: {time.time() - t0:.2f}s")
                case_id = data_ID[0]
                data_WSI = data_WSI.to(self.device)
                _, attn = model(data_WSI, return_attn=True)
                attn = attn.cpu()
                self.att_score[case_id] = attn

    def find_best_ckpt(self):
        filename_best = glob.glob(os.path.join(self.results_dir, "*.pth.tar"))
        assert len(filename_best) == 1
        self.filename_best = filename_best[0]


def normalize_case_id(case_id):
    case_id = case_id[0]
    if isinstance(case_id, torch.Tensor):
        return str(case_id.item())
    else:
        return str(case_id)