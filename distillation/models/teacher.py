from models import get_model
import torch.nn as nn


class Teacher(nn.Module):
    def __init__(self, config, rank):
        super(Teacher, self).__init__()
        teacher_config = config['model']['teacher']
        self.teachers = {teacher_name: get_model(teacher_name, rank) for teacher_name in teacher_config.keys()}

    def forward(self, x):
        outputs = {k: model(x[k]) for k, model in self.teachers.items()}
        return outputs
