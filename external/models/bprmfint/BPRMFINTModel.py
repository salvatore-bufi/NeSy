"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

import torch
import numpy as np
import random
import torch.nn.functional as F


class BPRMFINTModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 nint,
                 t,
                 lc,
                 random_seed,
                 name="BPRMFINT",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.nint = nint

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)
        self.Gr = torch.nn.Embedding(self.nint, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gr.weight)
        self.Gr.to(self.device)

        self.t = t
        self.lc = lc

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def info_nce_loss(self, temp=0.1):
        """
        InfoNCE loss to encourage distinct embeddings across rows.
        Args:
        - self.Gr.weight: matrix of embeddings (each row is an embedding).
        - temp: temp scaling factor (typically between 0.07 and 0.2).

        Returns:
        - The InfoNCE loss.
        """
        n = self.Gr.weight.size(0)  # Number of embeddings (rows)

        # Normalize the embeddings to unit length
        normed_embeddings = F.normalize(self.Gr.weight, p=2, dim=1)

        # Compute pairwise cosine similarities
        cosine_similarities = torch.matmul(normed_embeddings, normed_embeddings.t()) / temp

        # Create labels: the diagonal is the correct "positive" (same row is its own positive)
        labels = torch.arange(n).to(self.Gr.weight.device)

        # Compute InfoNCE loss (cross-entropy between the diagonal and off-diagonal elements)
        loss = F.cross_entropy(cosine_similarities, labels)

        return loss
    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        int = torch.softmax(self.Gr.weight @ self.Gi.weight.T, dim=1) @ self.Gi.weight
        u_int = torch.softmax(gamma_u @ int.T, dim=1) @ int

        gamma_u = gamma_u + u_int

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        gamma_u = self.Gu.weight[start:stop].to(self.device)
        int = torch.softmax(self.Gr.weight @ self.Gi.weight.T, dim=1) @ self.Gi.weight
        u_int = torch.softmax(gamma_u @ int.T, dim=1) @ int
        gamma_u = gamma_u + u_int
        return torch.matmul(gamma_u.to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        reg_loss = self.l_w * (0.5) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]

        reg_loss += self.l_w * (0.5) * self.Gr.weight.norm(2).pow(2)

        loss += reg_loss

        loss +=  self.lc * self.info_nce_loss(temp=self.t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
