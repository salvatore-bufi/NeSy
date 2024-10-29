from abc import ABC
import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn


class RBRSOPPOSITEModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 learning_rate: float,
                 embed_k: int,
                 l_w: int,
                 random_seed: int,
                 epsilon: float,
                 name="RBRSOPPOSITE",
                 **kwargs):
        super().__init__()

        # Set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k  # Embedding dimension for each rule
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.epsilon = epsilon  # Small constant to prevent log(0)

        # Initialize embeddings for users and items
        # For users, we have n_rules different embeddings (one for each rule)
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gu.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)

        self.weight = nn.Parameter(
            nn.init.normal(torch.empty((self.num_users, 1)), mean=0.0, std=1.0), requires_grad=True
        )

        # Item embeddings
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gi.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)



    def disjunction(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        expr = torch.log(1 - selector * selected + epsilon)
        log_sum = torch.sum(expr, dim=1)
        return 1 - (-1 / (-1 + log_sum))

    def conjunction_rule(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        """
        * This function performs a smooth AND-like operation on the tensors selector and selected. It multiplies the selector tensor by the complement of the selected tensor, then applies a logarithmic transformation to avoid numerical issues, sums these logs across dimensions, and returns a score between 0 and 1. The output represents the result of a smooth AND operation, with higher values indicating stronger conjunction (AND).
        """


        # Compute the element-wise terms
        expr = 1 - selector * (1 - selector) + epsilon  # epsilon prevents log(0)

        # Take the logarithm of each term
        log_expr = torch.log(expr)

        # Sum over the embedding dimensions
        sum_log_expr = torch.sum(log_expr, dim=1)

        # Compute the final AND function value
        res = -1.0 / (-1.0 + sum_log_expr)
        return res

    def forward(self, inputs, **kwargs):
        users, items = inputs
        batch_size = users.shape[0]

        # Get user embeddings and reshape for rules
        gu = self.Gu(torch.tensor(users))
        gu_opposite = -1 * gu


        # Get item embeddings
        gamma_i = self.Gi(torch.tensor(items)) # Shape: [batch_size, embed_k]

        # Compute the score of each rule using MF
        and_scores = []

        # Positive
        a, b = torch.sigmoid(self.weight[users]), 1 - torch.sigmoid(self.weight[users])
        score_and_one = a * torch.sum(gu * gamma_i, -1)
        score_and_two = b * torch.sum(gu_opposite * gamma_i, -1)
        and_scores.append(score_and_one.unsqueeze(1))
        and_scores.append(score_and_two.unsqueeze(1))

        # Concatenate and compute disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(and_scores, dim=1))  # Shape: [batch_size, n_rules]
        xui = self.disjunction(1.0, and_scores_tensor)  # Shape: [batch_size]

        return xui, gu, gamma_i


    def predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)
        gu_opposite = -1 * gu
        selector = 1.0

        gamma_i = self.Gi.weight.to(self.device)  # Shape: [num_items, embed_k]

        and_scores = []
        # Positive
        a, b = torch.sigmoid(self.weight[start:stop]), 1 - torch.sigmoid(self.weight[start:stop])
        score_and_one = a * torch.matmul(gu, torch.transpose(gamma_i, 0, 1))
        score_and_two = b * torch.matmul(gu_opposite, torch.transpose(gamma_i, 0, 1))
        and_scores.append(score_and_one.unsqueeze(0))
        and_scores.append(score_and_two.unsqueeze(0))



        # Combine scores using disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(and_scores, dim=0))  # [n_rules, num_users, num_items]
        # 1 - selector * selected + epsilon, selector, can be a trainable array
        expr = 1 - selector * and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]

        return final_scores


    def train_step(self, batch):
        user, pos, neg = batch


        # Positive items
        xui_pos, gu_pos, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        # Negative items
        xui_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        # BPR Loss V2
        # Compute the difference between positive and negative scores
        diff_scores = xui_pos - xui_neg  # Shape: (batch_size,)
        # Compute the BPR loss using the numerically stable softplus function
        loss = torch.nn.functional.softplus(-diff_scores).mean()

        # ----------------- Regularization Loss L2 -- Non va bene - sicuro da cambiare
        reg_loss = self.l_w * (
                gu_pos.norm(2).pow(2) +
                gamma_i_pos.norm(2).pow(2) +
                gamma_i_neg.norm(2).pow(2)
        ) / user.shape[0]


        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
