import torch.nn as nn


class DistillationLoss(nn.Module):
    def __init__(self, config):
        super(DistillationLoss, self).__init__()
        loss_config = config['loss']

        loss_type = loss_config['type']
        if loss_type == "L2":
            self.criterion = nn.MSELoss()
        elif loss_type == "L1":
            self.criterion = nn.L1Loss()
        elif loss_type == "SmoothL1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, student_features, teacher_features):
        loss = self.criterion(student_features, teacher_features)
        return loss


class DistillationLossMulti(nn.Module):
    def __init__(self, config):
        super(DistillationLossMulti, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = {k: v['weight'] for k, v in config['model']['teacher'].items()}

    def forward(self, student_features, teacher_features):
        loss_dict = {k: self.criterion(student_features[k], v) for k, v in teacher_features.items()}
        loss = sum(self.weights[k] * loss_dict[k] for k in teacher_features.keys())
        return loss, loss_dict
