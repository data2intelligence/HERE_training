import torch
import torch.nn as nn
import torch.nn.functional as F
from common import BACKBONE_DICT, CLASSIFICATION_DICT, REGRESSION_LIST


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=-1, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout: float
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # 1 x num_patches x 256
        b = self.attention_b(x)  # 1 x num_patches x 256
        A = a.mul(b)  # 1 x num_patches x 256
        A = self.attention_c(A)  # N x n_tasks, num_patches x 512
        return A, x


class AttentionModel(nn.Module):
    def __init__(self, backbone='PLIP', dropout=0.25):
        super().__init__()

        self.attention_net = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        ])
        self.rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        classifiers = {}
        for k, labels in CLASSIFICATION_DICT.items():
            classifiers[k] = nn.Linear(256, len(labels))
        self.classifiers = nn.ModuleDict(classifiers)
        regressors = {}
        for k in REGRESSION_LIST:
            regressors[k] = nn.Linear(256, 1)
        self.regressors = nn.ModuleDict(regressors)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_features=False, attention_only=False):
        x_path = x.squeeze(0)  # 1 x num_patches x 512  1 x 10000 x 512 --> all 256x256 pacthces

        A, h = self.attention_net(x_path)  # num_patches x num_tasks, num_patches x 512
        A = torch.transpose(A, 1, 0)  # num_tasks x num_patches
        # A_raw = A  # 1 x num_patches
        if attention_only:
            return {'A_raw': A}

        results_dict = {}
        A = F.softmax(A, dim=1)  # num_tasks x num_patches, normalized
        h = torch.mm(A, h)  # A: num_tasks x num_patches, h_path: num_patches x 256  --> num_tasks x 256
        if return_features:
            results_dict['global_feat'] = h
            results_dict['A'] = A
        h = self.rho(h)

        for k, classifier in self.classifiers.items():
            logits_k = classifier(h[0].unsqueeze(0))
            results_dict[k + '_logits'] = logits_k

        for k, regressor in self.regressors.items():
            values_k = regressor(h[0].unsqueeze(0)).squeeze(1)
            results_dict[k + '_logits'] = values_k

        return results_dict
