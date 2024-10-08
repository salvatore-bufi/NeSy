from abc import ABC
import torch
import numpy as np
import random
import torch.nn.functional as F


class RBRSModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 learning_rate: float,
                 embed_k: int,
                 l_w: int,
                 random_seed: int,
                 n_rules: int,
                 epsilon: float,
                 l_rc: float,
                 name="RBRS",
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
        self.n_rules = n_rules  # Number of conjunction rules
        self.epsilon = epsilon  # Small constant to prevent log(0)
        self.l_rc = l_rc

        # Initialize embeddings for users and items
        # For users, we have n_rules different embeddings (one for each rule)
        self.Gu = torch.nn.Embedding(self.num_users, self.n_rules * self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gu.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)

        # Item embeddings
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gi.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def conjunction_rule(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # Compute the conjunction (AND) function
        # u and i are tensors of shape [batch_size, embed_k]
        # Return tensor of shape [batch_size], i.e. u.shape[0] == i.shape[0]
        expr = 1 - u * (1 - i) + self.epsilon
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=1)
        res = -1.0 / (-1.0 + sum_log_expr)
        return res

    def disjunction_rule(self, and_rules: torch.Tensor) -> torch.Tensor:
        # Compute the disjunction (OR) function
        # and_rules is a tensor of shape [batch_size, n_rules], i.e. and_rules.shape[0] == batch_size, and_rules.shape[1] == n_rules
        expr = 1 - and_rules + self.epsilon
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=1)
        res = 1 - (-1.0 / (-1.0 + sum_log_expr))
        return res

    def forward(self, inputs, **kwargs):
        users, items = inputs
        batch_size = users.shape[0]

        # Get user embeddings and reshape for rules
        gu = self.Gu.weight[users]
        gu = gu.view(batch_size, self.n_rules, self.embed_k)  # 3 dim: 1st_dim = user_id, 2nd_dim = rule_no, 3d_dim = embedding of user-rule
        # Ensure embeddings are in [0,1]

        # Get item embeddings
        gamma_i = self.Gi.weight[items]  # Shape: [batch_size, embed_k]

        # Compute conjunctions for each rule
        and_scores = []
        for r in range(self.n_rules):
            gamma_u_r = gu[:, r, :]  # User embedding for rule r
            # score_and = self.conjunction_rule(u=gamma_u_r, i=gamma_i)  # Shape: [batch_size]
            score_and = torch.sum(gamma_u_r * gamma_i, -1)
            and_scores.append(score_and.unsqueeze(1))  # Shape: [batch_size, 1]

        # Concatenate and compute disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(and_scores, dim=1))  # Shape: [batch_size, n_rules]
        xui = self.disjunction_rule(and_scores_tensor)  # Shape: [batch_size]

        return xui, gu, gamma_i


    def predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)  # Shape: [num_users, n_rules * embed_k]
        gu = gu.view(gu.shape[0], self.n_rules, self.embed_k)

        gamma_i = self.Gi.weight.to(self.device)  # Shape: [num_items, embed_k]

        # Compute scores for each user and item
        scores = []
        for r in range(self.n_rules):
            gu_r = gu[:, r, :]  # [num_users, embed_k]
            # Broadcasting over items
            and_score = torch.matmul(gu_r, torch.transpose(gamma_i, 0, 1)) # [num_users, num_items]
            scores.append(and_score.unsqueeze(0))  # [1, num_users, num_items]

        # Combine scores using disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(scores, dim=0))  # [n_rules, num_users, num_items]
        expr = 1 - and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]

        return final_scores

    def dissimilarity_loss(self,  margin=0.5):
        r0, r1 = torch.split(self.Gu.weight, [self.embed_k, self.embed_k], 1)
        r0_normalized = F.normalize(r0, p=2, dim=1)
        r1_normalized = F.normalize(r1, p=2, dim=1)
        similarities = torch.sum(r0_normalized * r1_normalized, dim=1)
        loss = F.relu(similarities + margin).mean()
        return loss

    def mutual_information_loss(self, temperature=0.2):
        """
        Info NCE Loss
        Minimizing the mutual information between u[j, :] and i[j, :] ensures that knowing one embedding provides no
        information about the other, promoting semantic independence.
        Minimizes mutual information between u and i using InfoNCE loss.

        Args:
            u (Tensor): Embedding matrix u of shape (N, M).
            i (Tensor): Embedding matrix i of shape (N, M).
            temperature (float): Scaling factor.

        Returns:
            Tensor: Mutual information loss.
        """
        r0, r1 = torch.split(self.Gu.weight, [self.embed_k, self.embed_k], 1)
        batch_size = r0.size(0)
        r0_normalized = F.normalize(r0, p=2, dim=1)
        r1_normalized = F.normalize(r1, p=2, dim=1)

        # Compute similarity matrix
        similarities = torch.matmul(r0_normalized, r1_normalized.T)  # Shape: (N, N)
        similarities = similarities / temperature

        # Labels for InfoNCE (diagonal elements are positives)
        labels = torch.arange(batch_size).to(self.device)

        # Cross-entropy loss
        loss = F.cross_entropy(similarities, labels)

        return loss

    def vectorized_forward(self, inputs, **kwargs):
        users, items = inputs
        batch_size = users.shape[0]

        # Get user embeddings and reshape for rules
        gu = self.Gu.weight[users]
        gu = gu.view(batch_size, self.n_rules, self.embed_k)

        # Get item embeddings
        gamma_i = self.Gi.weight[items]

        # Compute conjunctions for each rule without the for-loop
        score_and = torch.sum(gu * gamma_i.unsqueeze(1), dim=-1)

        # Apply sigmoid activation
        and_scores_tensor = torch.sigmoid(score_and)
        xui = self.disjunction_rule(and_scores_tensor)  # Shape: [batch_size]

        return xui, gu, gamma_i

    def vectorized_predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)  # Shape: [num_users, n_rules * embed_k]
        gu = gu.view(gu.shape[0], self.n_rules, self.embed_k)  # Shape: [num_users, n_rules, embed_k]

        gamma_i = self.Gi.weight.to(self.device)  # Shape: [num_items, embed_k]

        # Compute scores for each user and item without a for-loop
        # Using torch.einsum to compute all scores at once
        # Compute [num_users, n_rules, num_items]
        and_scores = torch.einsum('unl,il->uni', gu, gamma_i)  # [num_users, n_rules, num_items]
        # Bring n_rules to the first dimension to match original shape
        and_scores = and_scores.permute(1, 0, 2)  # [n_rules, num_users, num_items]

        # Apply the sigmoid function
        and_scores_tensor = torch.sigmoid(and_scores)  # [n_rules, num_users, num_items]

        # Proceed with the rest of the computation
        expr = 1 - and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules: [num_users, num_items]
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]

        return final_scores


    def train_step(self, batch):
        user, pos, neg = batch

        # # Positive items
        # xui_pos, gu_pos, gamma_i_pos = self.forward(inputs=(user[:,0], pos[:, 0]))
        # # Negative items
        # xui_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        # Positive items
        xui_pos, gu_pos, gamma_i_pos = self.vectorized_forward(inputs=(user[:, 0], pos[:, 0]))
        # Negative items
        xui_neg, _, gamma_i_neg = self.vectorized_forward(inputs=(user[:, 0], neg[:, 0]))



        # BPR Loss
        # loss = -torch.mean(torch.nn.functional.logsigmoid(xui_pos - xui_neg))

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


        # ------------------ Rule Indipendence
        loss += self.l_rc * self.dissimilarity_loss()
        # loss += self.l_rc * self.mutual_information_loss()

        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
