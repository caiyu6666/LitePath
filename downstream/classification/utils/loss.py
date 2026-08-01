import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def define_loss(args):
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "aps":
        criterion = ScoreLoss(args.temperature)
    else:
        raise NotImplementedError
    return criterion


class ScoreLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ScoreLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        assert student_logits.shape == teacher_logits.shape, f"student_logits and teacher_logits must have the same shape: {student_logits.shape} != {teacher_logits.shape}"
        soft_ce = self.soft_cross_entropy(student_logits, teacher_logits)
        return soft_ce

    def soft_cross_entropy(self, student_logits, teacher_logits):
        p_t = F.softmax(teacher_logits / self.temperature, dim=0)
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=0)
        soft_ce = -torch.sum(p_t * log_p_s)
        return soft_ce
