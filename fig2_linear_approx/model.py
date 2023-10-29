import torch
import numpy as np

def log_clip(x):
    return torch.log(torch.clamp(x, 1e-10, None))


class LogisticRegression(torch.nn.Module):
    def __init__(self, weight_decay, is_multi=False):
        super(LogisticRegression, self).__init__()
        self.is_multi = is_multi
        # self.wd = torch.FloatTensor([weight_decay]).cuda()
        if self.is_multi:
            self.w = torch.nn.Parameter(torch.zeros([10, 784], requires_grad=True))
        else:
            self.w = torch.nn.Parameter(torch.zeros([784], requires_grad=True))

    def forward(self, x):
        if self.is_multi:
            logits = torch.matmul(x, self.w.T)
        else:
            logits = torch.matmul(x, torch.reshape(self.w, [-1, 1]))
        return logits

    def loss(self, logits, y, train=True):
        if self.is_multi:
            criterion = torch.nn.CrossEntropyLoss()
            # set dtype to float
            y = y.type(torch.FloatTensor)
            loss = criterion(logits, y.long())
        else:
            preds = torch.sigmoid(logits)

            if train:
                loss = -torch.mean(
                    y * log_clip(preds) + (1 - y) * log_clip(1 - preds))  # + torch.norm(self.w, 2) * self.wd
            else:
                loss = -torch.mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds))

        return loss
