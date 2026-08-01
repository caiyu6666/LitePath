import argparse
import json
import os


SPLIT_DIR = "dataset_split"
SPLIT_CSV = {
    "LUAD_survival": os.path.join(SPLIT_DIR, "LUAD_survival_5Fold.xlsx"),
    "LUSC_survival": os.path.join(SPLIT_DIR, "LUSC_survival_5Fold.xlsx"),
    "CRC_survival": os.path.join(SPLIT_DIR, "CRC_survival_5Fold.xlsx"),
    "HNSC_survival": os.path.join(SPLIT_DIR, "HNSC_survival_5Fold.xlsx"),
    "SKCM_survival": os.path.join(SPLIT_DIR, "SKCM_survival_5Fold.xlsx"),
}


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file", default=None)
    parser.add_argument("--feature", type=str, help="which feature extractor to use")
    parser.add_argument("--k_start", type=int, default=-1, help="start fold")
    parser.add_argument("--k_end", type=int, default=-1, help="end fold")
    parser.add_argument("--pt_roots", type=str, help="JSON string mapping dataset names to pt root directories.")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="ABMIL", help="type of model")
    parser.add_argument("--study", type=str, help="used dataset")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function (default: ce)")

    args = parser.parse_args()

    args.excel_file = SPLIT_CSV[args.study]

    try:
        args.pt_roots = json.loads(args.pt_roots)
    except json.JSONDecodeError as exc:
        raise ValueError("--pt_roots must be a valid JSON object string.") from exc
    if not isinstance(args.pt_roots, dict):
        raise ValueError("--pt_roots must decode to a dictionary of dataset names to paths.")
    return args


def aps_parse_args():
    parser = argparse.ArgumentParser(description="configurations for APS survival prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file", default=None)
    parser.add_argument("--feature", type=str, help="which feature extractor to use")
    parser.add_argument("--k_start", type=int, default=-1, help="start fold")
    parser.add_argument("--k_end", type=int, default=-1, help="end fold")
    parser.add_argument("--pt_roots", type=str, help="JSON string mapping dataset names to pt root directories.")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--aps_index", type=int, default=0, help="index of aps shallow block")
    parser.add_argument("--model", type=str, default="ABMIL", help="type of MIL model")
    parser.add_argument("--study", type=str, help="used dataset")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for softmax")
    parser.add_argument("--selected_num", type=int, default=2000, help="number of selected features")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=100, help="maximum number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="aps", help="APS loss function")

    args = parser.parse_args()
    args.excel_file = SPLIT_CSV[args.study]

    try:
        args.pt_roots = json.loads(args.pt_roots)
    except json.JSONDecodeError as exc:
        raise ValueError("--pt_roots must be a valid JSON object string.") from exc
    if not isinstance(args.pt_roots, dict):
        raise ValueError("--pt_roots must decode to a dictionary of dataset names to paths.")
    return args
