import argparse


SPLIT_CSV = {
    # Public
    "BRACS-3": "dataset_csv/BRACS-3.csv",
    "BRACS-7": "dataset_csv/BRACS-7.csv",
    "CAMELYON": "dataset_csv/Camelyon.csv",
    "NSCLC": "dataset_csv/NSCLC.csv",
    "LUAD_EGFR": "dataset_csv/LUAD_EGFR.csv",
    "LUAD_TP53": "dataset_csv/LUAD_TP53.csv",
    "COAD_READ_molecular": "dataset_csv/COAD_READ_molecular_subtyping.csv",
    # ......
}


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--root", type=str, help="path to root directory")
    parser.add_argument("--csv_file", type=str, help="path to csv file", default=None)
    parser.add_argument("--feature", type=str, help="which feature extractor to use")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--aggregator", type=str, default="", metavar="PATH", help="path to aggregator checkpoint")

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
    parser.add_argument("--loss", type=str, default="ce", help="slide-level classification loss function (default: ce)")
    args = parser.parse_args()

    args.csv_file = SPLIT_CSV[args.study]

    return args


def aps_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--root", type=str, help="path to root directory")
    parser.add_argument("--csv_file", type=str, help="path to csv file", default=None)
    parser.add_argument("--feature", type=str, help="which feature extractor to use")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--aggregator", type=str, default="", metavar="PATH", help="path to aggregator checkpoint")
    parser.add_argument("--aps_index", type=int, default=0, help="index of aps")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="ABMIL", help="type of model")
    parser.add_argument("--study", type=str, help="used dataset")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for softmax")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=100, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="aps", help="slide-level classification loss function (default: ce)")
    args = parser.parse_args()

    args.csv_file = SPLIT_CSV[args.study]

    return args
