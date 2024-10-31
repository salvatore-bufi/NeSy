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

class RSLOGIC2Model(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 ui: np.array,
                 learning_rate: float,
                 embed_k: int,
                 l_w: float,
                 random_seed: int,
                 name="RSLOGIC2",
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

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # Int extraction
        self.int_layer1 = torch.nn.Linear(2 * self.embed_k, self.embed_k)
        self.int_layer2 = torch.nn.Linear(self.embed_k, self.embed_k)

        self.ui = ui

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def interaction_repr(self, user, item):
        ui_vector = torch.cat((user, item), dim=1)
        ui_vector = F.leaky_relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector



    def interaction_repr_history(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        ui_vector = torch.cat((user.view(1, -1).expand(items.size(0), -1), items), dim=1)
        ui_vector = F.leaky_relu(self.int_layer1(ui_vector))
        ui_vector = self.int_layer2(ui_vector)
        return ui_vector

    def extract_int_from_user_history(self, user: np.array, ui: torch.Tensor) -> torch.Tensor:
        gu_list = []
        for u in user:
            gu = self.Gu.weight[u]
            user_history = ui[:, ui[0, :] == u][1, :] ## torch tensor of lenght == number_of_items the user interacted
            user_items = torch.mean(self.interaction_repr_history(gu, self.Gi(user_history)), dim=0)  # return a tensor of shape (n_interactions, dim_emb)
            gu_list.append(user_items.unsqueeze(0))
        gu_star = torch.cat(gu_list, dim=0)
        return gu_star

    def forward(self, inputs, **kwargs):
        users, items = inputs

        gu_star = self.extract_int_from_user_history(users, self.ui)
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        gui = self.interaction_repr(gamma_u, gamma_i)

        xui = torch.sum(gu_star * gui, 1)

        return xui, gu_star, gamma_i

    def predict(self, start: int, stop: int, **kwargs) -> torch.Tensor:
        """
        Predicts scores for all items for users in the range [start, stop).

        Args:
            start (int): Starting index of users.
            stop (int): Stopping index of users.
            **kwargs: Additional arguments (if any).

        Returns:
            torch.Tensor: Prediction scores of shape (stop - start, num_items).
        """
        # Generate user indices
        users = torch.arange(start, stop, device=self.device)
        num_users = users.size(0)

        # Generate item indices
        items = torch.arange(self.num_items, device=self.device)
        num_items = items.size(0)

        # Retrieve user embeddings
        gamma_u = self.Gu(users)  # Shape: (num_users, embed_k)

        # Retrieve all item embeddings
        gamma_i = self.Gi(items)  # Shape: (num_items, embed_k)

        # Expand user and item embeddings for pairwise computation
        # gamma_u_exp: (num_users, num_items, embed_k)
        # gamma_i_exp: (num_users, num_items, embed_k)
        gamma_u_exp = gamma_u.unsqueeze(1).expand(-1, num_items, -1)
        gamma_i_exp = gamma_i.unsqueeze(0).expand(num_users, -1, -1)

        # Compute interaction representations for all user-item pairs
        # First, reshape to (num_users * num_items, embed_k) for batch processing
        gamma_u_flat = gamma_u_exp.contiguous().view(-1, self.embed_k)
        gamma_i_flat = gamma_i_exp.contiguous().view(-1, self.embed_k)

        # Compute interaction representation using the interaction_repr method
        gui_flat = self.interaction_repr(gamma_u_flat, gamma_i_flat)  # Shape: (num_users * num_items, embed_k)

        # Reshape back to (num_users, num_items, embed_k)
        gui = gui_flat.view(num_users, num_items, self.embed_k)

        # Aggregate user history to get gu_star for each user
        # Convert user indices to NumPy for compatibility with extract_int_from_user_history
        users_np = users.cpu().numpy()
        gu_star = self.extract_int_from_user_history(users_np, self.ui).to(self.device)  # Shape: (num_users, embed_k)

        # Expand gu_star to match the shape of gui for element-wise multiplication
        gu_star_exp = gu_star.unsqueeze(1).expand(-1, num_items, -1)  # Shape: (num_users, num_items, embed_k)

        # Compute the dot product between gu_star and gui for each user-item pair
        # This results in the final prediction scores
        xui = torch.sum(gu_star_exp * gui, dim=2)  # Shape: (num_users, num_items)

        return xui

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        reg_loss = self.l_w * (1 / 2) * (self.Gu.weight.norm(2).pow(2) +
                                         self.Gi.weight.norm(2).pow(2)) / (self.Gu.weight.shape[0] + self.Gi.weight.shape[0])
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
