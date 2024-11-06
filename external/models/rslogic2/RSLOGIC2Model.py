import torch
import numpy as np
import random
from abc import ABC
from .nesy_disj import NeuroSymbolicDisjunction
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
                 **kwargs):
        super().__init__()

        # Set seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.logic_w = 0.05

        self.device = 'cpu'

        # Embeddings for users and items
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k, device=self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k, device=self.device)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)

        # Convert ui to tensor on device
        self.ui = torch.tensor(ui, device=self.device, requires_grad=False)
        self.user_history = {int(u): self.ui[:, self.ui[0, :] == u][1, :] for u in self.ui[0, :].unique()}

        self.method = 'softmax'
        self.disjunction = NeuroSymbolicDisjunction(method=self.method)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def like_history(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(u, i.T)).unsqueeze(0)

    def antecedent(self, x: torch.Tensor) -> torch.Tensor:
        return 1. - x

    def remove_items(self, u, i):
        # Convert u and i to lists to loop through
        u = u.tolist()
        i = i.tolist()

        # Use a dictionary to track items to remove for each user
        removal_dict = {}
        for user, item in zip(u, i):
            if user in removal_dict:
                removal_dict[user].append(item)
            else:
                removal_dict[user] = [item]

        # Create a new dictionary for updated user_history
        new_user_history = {}
        for user, items in self.user_history.items():
            if user in removal_dict:
                # Convert the list of items to be removed to a tensor for fast filtering
                removal_tensor = torch.tensor(removal_dict[user])
                # Keep only items not in the removal list
                filtered_items = items[~torch.isin(items, removal_tensor)]
                new_user_history[user] = filtered_items
            else:
                # If no items need to be removed, retain the original tensor
                new_user_history[user] = items

        return new_user_history

    def premise_user_history(self, u: int, user_history: dict = None) -> torch.Tensor:
        if self.user_history == None:
            user_history = self.user_history[u]
        gu = self.Gu.weight[u]
        gi = self.Gi(user_history)
        rule_antecedent = self.antecedent(self.like_history(gu, gi))
        return self.disjunction.forward(rule_antecedent)
    def calculate_batch_premise(self, users: np.array, ui = None):
        if ui == None:
            u_premis_list = [self.premise_user_history(u).unsqueeze(0) for u in users]
        else:
            u_premis_list = [self.premise_user_history(u, ui[u]).unsqueeze(0) for u in users]
        return torch.cat(u_premis_list, dim=0)

    def forward_logic(self, inputs, **kwargs):
        users_premises, scores = inputs
        users_premises = users_premises
        ui = torch.sigmoid(scores).unsqueeze(1)
        xui = self.disjunction.forward(torch.cat((ui, users_premises), dim=1))
        return xui

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

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = self.Gu.weight[users]
        gamma_i = self.Gi.weight[items]
        xui = torch.sum(gamma_u * gamma_i, dim=1)
        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop], self.Gi.weight.T)

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        loss = 0
        bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        loss += bpr_loss

        user_history = self.remove_items(user[:, 0], pos[:, 0])
        premise = self.calculate_batch_premise(user[:, 0], user_history)
        logic_score_pos = self.forward_logic(inputs=(premise, xu_pos))
        logic_score_neg = self.forward_logic(inputs=(premise, xu_neg))

        logic_loss = F.binary_cross_entropy(logic_score_pos, torch.ones(user.shape[0])) + F.binary_cross_entropy(logic_score_neg, torch.zeros(user.shape[0]))
        # logic_loss = -torch.mean(torch.nn.functional.logsigmoid(logic_score_pos - logic_score_neg))
        # print(f"\nBPR LOSS: {loss} \t LOGIC LOSS: {logic_loss}")
        # loss += self.logic_w * logic_loss
        # print(f"\n BPR: {bpr_loss} \t LOGIC: {logic_loss} ")
        loss += self.logic_w * (0.5) * logic_loss

        reg_loss = self.l_w * 0.5 * (gamma_u.norm(2).pow(2) +
                                     gamma_i_pos.norm(2).pow(2) +
                                     gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
