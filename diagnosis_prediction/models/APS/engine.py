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
    def __init__(self, args, results_dir, fold, aps_results_dir):
        self.args = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.num_folds > 1:
            self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        else:
            self.results_dir = results_dir

        self.aps_results_dir = aps_results_dir

        self.val_scores = None
        self.test_scores = None if self.args.num_folds > 1 else dict()
        self.filename_best = None
        self.mil_filename_best = None
        self.best_epoch = 0
        self.best_ckpt = None
        self.early_stop = 0
        self.att_score = {}

        # self.att_score_dir = att_score_dir
        # self.pred_att_score_dir = pred_att_score_dir
        # self.selected_feat_dir = selected_feat_dir

    def learning(self, model, loaders, criterion, optimizer, scheduler):
        print(">>>")
        print(">>>")
        print(f"**************************** start learning APS *****************************")
        if os.path.exists(os.path.join(self.aps_results_dir, "aps_model_best.pth.tar")):
            print(f"==== aps_model_best.pth.tar already exists in {self.aps_results_dir}. Skip...")
            self.filename_best = os.path.join(self.aps_results_dir, f"aps_model_best.pth.tar")
            return

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(self.best_epoch, self.args.num_epoch):
            print(f"--------------------------------Epoch {epoch} / {self.args.num_epoch}--------------------------------")
            self.epoch = epoch
            train_loader, val_loader, test_loader, external_loaders = loaders
            # train
            train_loss = self.train(train_loader, model, criterion, optimizer)
            # evaluate
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


    def evaluate_with_selection_all_loaders(self, loaders, mil_model, selection):
        print("Selection: ", selection)

        if torch.cuda.is_available():
            mil_model = mil_model.cuda()

        self.mil_filename_best = self.find_best_mil_ckpt()
        state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=True)['state_dict']
        mil_model.load_state_dict(state_dict)
        print(f"=> load best mil model from {self.mil_filename_best}")

        test_loader, external_loaders = loaders
        test_scores, test_outputs = self.evaluate_with_selection(test_loader, mil_model, selection)

        external_scores, external_outputs = {}, {}
        if external_loaders is not None:
            for key, one_external_loader in external_loaders.items():
                external_scores[key], external_outputs[key] = self.evaluate_with_selection(one_external_loader, mil_model, selection)
        outputs = {'test': test_outputs}
        outputs.update(external_outputs)
        self.save_outputs(outputs)
        return test_scores, external_scores

    def evaluate_with_selection(self, eval_loader, mil_model, selection="50_1950"):
        if torch.cuda.is_available():
            mil_model = mil_model.cuda()
        if self.mil_filename_best is None:
            self.mil_filename_best = self.find_best_mil_ckpt()
            state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=True)['state_dict']
            mil_model.load_state_dict(state_dict)
            print(f"=> load best mil model from {self.mil_filename_best}")

        mil_model.eval()
        # selection: (APS_num, uniform_num)
        aps_num, uniform_num = selection.split("_")
        aps_num, uniform_num = int(aps_num), int(uniform_num)
        print(f"=> evaluate with selection: {aps_num} APS and {uniform_num} uniform")
        cases, slides, logits, labels, attns = [], [], [], [], []
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, (data_ID, data_Slide, data_WSI, data_Label, pred_score) in enumerate(eval_loader):
                if batch_idx % 50 == 0:
                    print(f"    {batch_idx}/{len(eval_loader)} time: {time.time() - t0:.2f}s")
                data_WSI = data_WSI.to(self.device)
                one_hot_label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
                pred_score = pred_score.squeeze()

                all_indices = torch.arange(data_WSI.shape[1])
                if aps_num+uniform_num >= data_WSI.shape[1]:
                    selected_indices = all_indices
                else:
                    uniform_indices = torch.linspace(0, data_WSI.shape[1]-1, steps=uniform_num).int()
                    remain_indices = torch.tensor([i for i in all_indices if i not in uniform_indices])
                    remain_scores = pred_score[remain_indices]
                    topk_idx_in_remain = torch.topk(remain_scores, k=aps_num).indices
                    topk_indices = remain_indices[topk_idx_in_remain]
                    selected_indices = torch.cat([uniform_indices, topk_indices])

                data_WSI = data_WSI[:, selected_indices]
                # print(data_WSI.shape)
                with torch.no_grad():
                    logit, attn = mil_model(data_WSI, return_attn=True)
                # results
                all_labels = np.row_stack((all_labels, one_hot_label.cpu().numpy()))
                all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))

                cases.append(normalize_case_id(data_ID))
                slides.append(data_Slide[0])
                logits.append(logit.cpu().numpy())
                labels.append(data_Label.item())
                attns.append(attn.cpu().numpy())

        # calculate metrics
        scores, _ = self.mil_metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        
        logits = np.concatenate(logits, axis=0)
        labels = np.array(labels)
        outputs = {'cases': cases, 'slides': slides, 'logits': logits, 'labels': labels, 'attns': attns}

        return scores, outputs

    def print_scores(self, scores):
        for k, v in scores.items():
            print(f"    {k}: {v:.4f}")
        print("---")

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(
            self.aps_results_dir,
            f"aps_model_best.pth.tar"
        )
        torch.save(state, self.filename_best)
        print("save best model {filename}".format(filename=self.filename_best))

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for idx, data_item in enumerate(data_loader):
            if idx % 50 == 0:
                print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
            data, att_score = data_item['shallow_feature'], data_item['att_score']
            data, att_score = data.to(self.device), att_score.to(self.device)

            outputs = model(data)
            try:
                loss = criterion(outputs.squeeze(), att_score.squeeze())
            except:
                print(f"{data_item['slide']}: shallow_feature {data.shape}, att_score {att_score.shape}, outputs {outputs.shape}")
                raise ValueError("Error in training")
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / len(data_loader)

    def validate(self, data_loader, model):
        metrics_dict = {'top10_recall': [], 'top250_recall': [], 'top500_recall': [], 'softmax_sum': [], 'reference_sum': []}
        model.eval()
        t0 = time.time()
        for idx, data_item in enumerate(data_loader):
            if idx % 50 == 0:
                print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
            data, att_score = data_item['shallow_feature'], data_item['att_score']
            data, att_score = data.to(self.device), att_score.to(self.device)
            outputs = model(data)
            metrics = self.metrics(outputs.squeeze(), att_score.squeeze())
            for k, v in metrics.items():
                metrics_dict[k].append(v)
        metrics_dict = {k: np.mean(v) for k, v in metrics_dict.items()}
        return metrics_dict

    def metrics(self, pred_score, att_score):
        assert pred_score.shape[0] == att_score.shape[0]
        N = pred_score.shape[0]
        # 1. 预测的 top-500 instance 索引
        pred_topk = torch.topk(pred_score, k=min(500, N)).indices  # [500]
        # 2. 标签 att_score 的 top-N 索引
        label_top10 = torch.topk(att_score, k=min(10, N)).indices
        label_top250 = torch.topk(att_score, k=min(250, N)).indices
        label_top500 = torch.topk(att_score, k=min(500, N)).indices

        top10_recall = recall(label_top10, pred_topk)
        top250_recall = recall(label_top250, pred_topk)
        top500_recall = recall(label_top500, pred_topk)

        # 4. 预测的 top-500 instance 中，真实 att_score 经 softmax 后的概率和
        att_score_softmax = F.softmax(att_score, dim=0)  # [N]
        softmax_sum = att_score_softmax[pred_topk].sum().item()
        reference_sum = att_score_softmax[label_top500].sum().item()

        return {'top10_recall': top10_recall, 'top250_recall': top250_recall, 'top500_recall': top500_recall, 'softmax_sum': softmax_sum, 'reference_sum': reference_sum}

    def evaluate_topk(self, mil_model, loader, k_list=[10, 50, 100, 200, 300, 500, 1000], save_dir=None, mode='top', key='val'):
        assert mode in ['top', 'uniform']
        k_list.append(-1) if -1 not in k_list else None
        if torch.cuda.is_available():
            mil_model = mil_model.cuda()
        self.mil_filename_best = self.find_best_mil_ckpt()
        mil_model.load_state_dict(torch.load(self.mil_filename_best, map_location=self.device)['state_dict'])
        print(f"=> load best mil model from {self.mil_filename_best}")

        file_name = os.path.join(save_dir, f"outputs_{key}_{mode}.pkl")
        if os.path.exists(file_name):
            print(f"==== {file_name} already exists. Load as initial outputs...")
            with open(file_name, "rb") as f:
                outputs_key = pickle.load(f)
        else:
            outputs_key = {}

        for k in k_list:
            if k == -1 and mode == 'uniform':
                continue
            
            if k in outputs_key.keys():
                print(f"==== {k} already exists. Skip...")
                continue
            # file_name = os.path.join(save_dir, f"outputs_val_{mode}{k}.pkl")
            # if os.path.exists(file_name):
            #     print(f"==== {file_name} already exists. Skip...")
            #     continue
            scores, outputs = self.evaluate_topk_one_loader(mil_model, loader, k=k, mode=mode, key=key)
            print(f"{mode}{k} {key} scores: {scores}")
            outputs_key[k] = outputs
            # with open(file_name, "wb") as f:
            #     pickle.dump(val_outputs, f)
            # print(f"Save val_outputs to {file_name}")
        
        with open(file_name, "wb") as f:
            pickle.dump(outputs_key, f)
        print(f"Save results to {file_name}")

    def evaluate_topk_one_loader(self, mil_model, dataloader, k=50, mode='top', key='val'):
        assert mode in ['top', 'uniform']
        mil_model.eval()
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(dataloader, desc=f"{mode} {k} {key}")
        else:
            # dataloader = dataloader
            print(f"-------------------------------{mode} {k} {key}-------------------------------")

        cases, slides, logits, labels, attns = [], [], [], [], []
        t0 = time.time()
        for batch_idx, (data_ID, data_Slide, data_WSI, data_Label) in enumerate(dataloader):
            if batch_idx % 20 == 0:
                print(f"    {batch_idx}/{len(dataloader)} time: {time.time() - t0:.2f}s")
            data_WSI = data_WSI.to(self.device)

            one_hot_label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            with torch.no_grad():
                attn = mil_model(data_WSI, return_attn=True, attn_only=True)
                if k != -1:  # 如果 k == -1，则使用所有patch，否则只使用前 k 个patch
                    select_num = min(k, attn.shape[1])
                    if mode == 'top':
                        select_idx = torch.topk(attn, k=select_num, dim=1).indices.squeeze()
                    elif mode == 'uniform':
                        select_idx = torch.linspace(0, attn.shape[1]-1, steps=select_num).int()
                    else:
                        raise ValueError(f"Invalid mode: {mode}")
                    data_WSI = data_WSI[:, select_idx]

                # print(data_WSI.shape)
                logit = mil_model(data_WSI)
            # results
            all_labels = np.row_stack((all_labels, one_hot_label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))

            cases.append(normalize_case_id(data_ID))
            slides.append(data_Slide[0])
            logits.append(logit.cpu().numpy())
            labels.append(data_Label.item())
            attns.append(attn.cpu().numpy())

        # calculate metrics
        scores, _ = self.mil_metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))

        logits = np.concatenate(logits, axis=0)
        labels = np.array(labels)
        outputs = {'cases': cases, 'slides': slides, 'logits': logits, 'labels': labels, 'attns': attns}
        return scores, outputs

    def infer_selection(self, model, loaders, score_only=True):
        print(">>>")
        print(">>>")
        print(f"************************* Infer APS to get estimated att_score *************************")
        pred_score_path = os.path.join(self.aps_results_dir, f"pred_score.pt")

        train_loader, val_loader, test_loader, external_loaders = loaders

        if os.path.exists(pred_score_path):
            pred_att_score = torch.load(pred_score_path)
            print(f"=> load pred_att_score from {pred_score_path}")
        else:
            pred_att_score = {}
            if torch.cuda.is_available():
                model = model.cuda()
            model.load_state_dict(torch.load(self.filename_best, map_location=self.device)['state_dict'])
            print(f"=> load best aps model from {self.filename_best}")

            train_pred_att_score = self.infer_pred_score_one_loader(model, train_loader, split='train')
            val_pred_att_score = self.infer_pred_score_one_loader(model, val_loader, split='val')
            test_pred_att_score = self.infer_pred_score_one_loader(model, test_loader, split='test')
            pred_att_score.update(train_pred_att_score)
            pred_att_score.update(val_pred_att_score)
            pred_att_score.update(test_pred_att_score)

            if external_loaders is not None:
                for key, one_external_loader in external_loaders.items():
                    external_pred_att_score = self.infer_pred_score_one_loader(model, one_external_loader, split=key)
                    pred_att_score.update(external_pred_att_score)
            torch.save(pred_att_score, pred_score_path)  # save full pred_att_score
            print(f"=> save pred_att_score to {pred_score_path}")

        if score_only:
            return

    def infer_pred_score_one_loader(self, model, data_loader, split='test'):
        print(f"=> infer {split} loader pred_score ... ")
        pred_att_score = {}
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, data_item in enumerate(data_loader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(data_loader)} time: {time.time() - t0:.2f}s")
                case_id, slide, data = data_item['case_id'], data_item['slide'], data_item['shallow_feature']
                data = data.to(self.device)
                outputs = model(data)
                case_id = normalize_case_id(case_id)
                slide = str(slide[0])
                key = f"{case_id}_{slide}"
                pred_att_score[key] = outputs.cpu()
        return pred_att_score

    def save_outputs(self, outputs):
        for key, value in outputs.items():
            file_name = os.path.join(self.aps_results_dir, f"outputs_{key}.pkl")
            with open(file_name, "wb") as f:
                pickle.dump(value, f)
            print(f"Save outputs to {file_name}")

    def mil_metrics(self, logits, labels, bootstrap=False):
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

    # ------------- Get the att_score of the best mil model -------------
    def infer_att_score(self, mil_model, mil_loaders):
        print(">>>")
        print(">>>")
        print(f"************************* Infer att_score of the best mil model *************************")
        save_path = os.path.join(self.aps_results_dir, "att_score.pt")
        if os.path.exists(save_path):
            print(f"==== att_score already exists in {save_path}. Skip...")
            return

        if torch.cuda.is_available():
            mil_model = mil_model.cuda()
        self.mil_filename_best = self.find_best_mil_ckpt()
        state_dict = torch.load(self.mil_filename_best, map_location=self.device, weights_only=True)['state_dict']
        mil_model.load_state_dict(state_dict)
        print(f"=> load best mil model from {self.mil_filename_best}")

        train_loader, val_loader, test_loader, external_loader = mil_loaders
        self.infer_att_score_one_loader(mil_model, train_loader, key='train')
        self.infer_att_score_one_loader(mil_model, val_loader, key='val')
        self.infer_att_score_one_loader(mil_model, test_loader, key='test')
        if external_loader is not None:
            for key, loader in external_loader.items():
                self.infer_att_score_one_loader(mil_model, loader, key=key)
        torch.save(self.att_score, save_path)
        print(f"=> save att_score to {save_path}")
        self.att_score = None  # clear memory

    def infer_att_score_one_loader(self, mil_model, mil_dataloader, key='train'):
        print(f"=> infer {key} loader attn score ... ")
        mil_model.eval()
        t0 = time.time()
        with torch.no_grad():
            for idx, (data_ID, data_Slide, data_WSI, data_Label) in enumerate(mil_dataloader):
                if idx % 50 == 0:
                    print(f"    {idx}/{len(mil_dataloader)} time: {time.time() - t0:.2f}s")
                
                data_WSI = data_WSI.to(self.device)
                _, attn = mil_model(data_WSI, return_attn=True)
                attn = attn.cpu()
                
                case_id = normalize_case_id(data_ID)
                data_Slide = str(data_Slide[0])
                key = f"{case_id}_{data_Slide}"

                if key in self.att_score.keys():
                    print(f"Warning: key {key} already exists")
                #     print(f"key {key} already exists")
                #     print(self.att_score.keys())
                #     raise ValueError(f"key {key} already exists")

                self.att_score[key] = attn

    def find_best_mil_ckpt(self):
        mil_filename_best = glob.glob(os.path.join(self.results_dir, "*.pth.tar"))
        assert len(mil_filename_best) == 1
        return mil_filename_best[0]


def normalize_case_id(case_id):
    case_id = case_id[0]
    if isinstance(case_id, torch.Tensor):
        return str(case_id.item())
    else:
        return str(case_id)


def recall(topk_true, pred_topk):
    # 预测的 top-200 中有多少是真实 top-K
    num_recalled = len(set(topk_true.tolist()) & set(pred_topk.tolist()))
    return num_recalled / len(topk_true)
