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


class RSLOGICModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 ui: torch.Tensor,
                 learning_rate: float,
                 embed_k: int,
                 l_w: float,
                 random_seed: int,
                 name="RSLOGIC",
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
        self.ui = ui

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)


        #Int extraction
        self.int_layer1 = torch.nn.Linear(2 * self.embed_k, self.embed_k)
        self.int_layer2 = torch.nn.Linear(self.embed_k, self.embed_k)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def extract_int_from_user_history(self, user: np.array):

        ''' user: np.array containing indices of the user'''
        gu_list = []
        for u in user:
            gu = self.Gu.weight[u]
            user_history = self.ui[:, self.ui[0, :] == u][1, :] ## torch tensor of lenght == number_of_items the user interacted
            user_items = torch.mean(self.interaction_int_history(gu, self.Gi(user_history)), dim=0)  # return a tensor of shape (n_interactions, dim_emb)
            gu_list.append(user_items.unsqueeze(0))
        gu_star = torch.cat(gu_list, dim=0)
        return gu_star



    def interaction_int(self, user, item):
        ui_vector = torch.cat((user, item), dim=1)
        ui_vector = F.relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector

    def interaction_int_history(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        ui_vector = torch.cat((user.view(1, -1).expand(items.size(0), -1), items), dim=1)
        ui_vector = F.relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector


    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop].to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
