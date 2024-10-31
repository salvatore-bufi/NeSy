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

        # U-I history after batch filtering
        self.ui_filtered = None


        #Int extraction
        self.int_layer1 = torch.nn.Linear(2 * self.embed_k, self.embed_k)
        self.int_layer2 = torch.nn.Linear(self.embed_k, self.embed_k)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def remove_batch_pairs(self, A, B):
        """
        Removes all (user, item) pairs from self.ui that are present in the batch (A, B).

        Parameters:
        - self.ui: Tensor of shape [2, num_interactions]
        - A: Tensor of shape [batch_size,] containing user indices
        - B: Tensor of shape [batch_size,] containing item indices

        Returns:
        - self.ui_filtered: Tensor of shape [2, num_remaining_interactions]
        """
        # Ensure self.ui, A, B are long tensors for integer operations
        A = torch.tensor(A).long()
        B = torch.tensor(B).long()

        # Determine the maximum user and item indices to create a unique encoding
        max_user = torch.max(self.ui[0]).item()
        max_item = torch.max(self.ui[1]).item()

        # Encode (user, item) pairs uniquely
        encoding_factor = max_item + 1  # Ensures unique encoding
        code_batch = A * encoding_factor + B  # Shape: [batch_size]
        code_int = self.ui[0] * encoding_factor + self.ui[1]  # Shape: [num_interactions]

        # Create a mask where self.ui pairs are NOT in the batch pairs
        mask = ~torch.isin(code_int, code_batch)

        # Apply the mask to filter out unwanted pairs
        ui_filtered = self.ui[:, mask]

        return ui_filtered

    def extract_int_from_user_history(self, user: np.array, ui: torch.Tensor) -> torch.Tensor:

        ''' user: np.array containing indices of the user
            ui: user_history [2, n_int]'''
        gu_list = []
        for u in user:
            gu = self.Gu.weight[u]
            user_history = ui[:, ui[0, :] == u][1, :] ## torch tensor of lenght == number_of_items the user interacted
            user_items = torch.mean(self.interaction_int_history(gu, self.Gi(user_history)), dim=0)  # return a tensor of shape (n_interactions, dim_emb)
            gu_list.append(user_items.unsqueeze(0))
        gu_star = torch.cat(gu_list, dim=0)
        return gu_star




    def interaction_int(self, user, item):
        ui_vector = torch.cat((user, item), dim=1)
        ui_vector = F.leaky_relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector

    def interaction_int_history(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        ui_vector = torch.cat((user.view(1, -1).expand(items.size(0), -1), items), dim=1)
        ui_vector = F.leaky_relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector

    def interaction_int_all_items(self, user: torch.Tensor) -> torch.tensor:
        items = self.Gi.weight
        ui_vector = torch.cat((user.view(1, -1).expand(items.size(0), -1), items), dim=1)
        ui_vector = F.leaky_relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector



    def forward(self, inputs, **kwargs):
        gu_star, user, items = inputs

        gamma_u = torch.squeeze(self.Gu.weight[user]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        gi = self.interaction_int(gamma_u, gamma_i)

        xui = torch.sum(gu_star * gi, 1)

        return xui, gu_star, gi

    def predict(self, start, stop, **kwargs):
        users = np.arange(start, stop)

        scores = []
        for u in users:
            gu_star = self.extract_int_from_user_history([u], self.ui)
            Gi_u = self.interaction_int_all_items(self.Gu.weight[u])
            u_score = torch.matmul(gu_star, torch.transpose(Gi_u, 0, 1))
            scores.append(u_score)
        scores = torch.cat(scores, 0)
        return scores


    def train_step(self, batch):
        user, pos, neg = batch

        batch_history = self.remove_batch_pairs(user[:, 0], pos[:, 0])
        gu_star = self.extract_int_from_user_history(user[:, 0], batch_history)

        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu_star, user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu_star, user[:, 0], neg[:, 0]))

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
