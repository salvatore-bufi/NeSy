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



    def disjunction_rule(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        expr = torch.log(1 - selector * selected + epsilon)
        log_sum = torch.sum(expr, dim=1)
        return 1 - (-1.0 / (-1.0 + log_sum))

    def conjunction_rule(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        """
        * This function performs a smooth AND-like operation on the tensors selector and selected. It multiplies the selector tensor by the complement of the selected tensor, then applies a logarithmic transformation to avoid numerical issues, sums these logs across dimensions, and returns a score between 0 and 1. The output represents the result of a smooth AND operation, with higher values indicating stronger conjunction (AND).
        """


        # Compute the element-wise terms
        expr = 1 - selector * (1 - selected) + epsilon  # epsilon prevents log(0)

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
        gu = self.Gu.weight[users]
        gu = gu.view(batch_size, self.n_rules, self.embed_k)  # 3 dim: 1st_dim = user_id, 2nd_dim = rule_no, 3d_dim = embedding of user-rule
        # Ensure embeddings are in [0,1]

        # Get item embeddings
        gamma_i = self.Gi.weight[items]  # Shape: [batch_size, embed_k]

        # Compute the score of each rule using MF
        and_scores = []
        for r in range(self.n_rules):
            gamma_u_r = gu[:, r, :]  # User embedding for rule r
            score_and = torch.sum(gamma_u_r * gamma_i, -1)
            and_scores.append(score_and.unsqueeze(1))  # Shape: [batch_size, 1]

        # Concatenate and compute disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(and_scores, dim=1))  # Shape: [batch_size, n_rules]
        xui = self.disjunction_rule(1.0, and_scores_tensor)  # Shape: [batch_size]

        return xui, gu, gamma_i


    def predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)  # Shape: [num_users, n_rules * embed_k]
        gu = gu.view(gu.shape[0], self.n_rules, self.embed_k)
        selector = 1.0

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
        # 1 - selector * selected + epsilon, selector, can be a trainable array
        expr = 1 - selector * and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
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

    def dissimilarity_loss_matrix(self, margin=0.5):
        # Assume self.Gu.weight has shape (N, M * self.embed_k)
        N = self.Gu.weight.size(0)
        M = self.n_rules  # Number of partitions, replace with your variable if different

        # Split the embeddings into M parts
        embeddings = self.Gu.weight.view(N, M, self.embed_k)  # Shape: (N, M, embed_k)

        # Normalize embeddings
        norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # Normalize along the embedding dimension

        # Compute pairwise similarities between embeddings of different parts
        # similarities: Tensor of shape (N, M, M), where similarities[n, i, j] = similarity between
        # the i-th and j-th embeddings for the n-th sample
        similarities = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))  # Shape: (N, M, M)

        # Exclude self-similarities (diagonal elements) and duplicate pairs
        batch_size = embeddings.size(0)
        idx = torch.triu_indices(M, M, offset=1)
        pairwise_similarities = similarities[:, idx[0], idx[1]]  # Shape: (N, num_pairs)

        # Compute the loss
        loss = F.relu(pairwise_similarities + margin).mean()  # Mean over all pairs and samples
        return loss

    def mutual_information_loss_vectorized(self, temperature=0.2):
        """
        Minimizes mutual information between embeddings using InfoNCE loss.
        """
        # Assume self.Gu.weight has shape (N, M * self.embed_k)
        N = self.Gu.weight.size(0)
        M = self.n_rules  # Number of partitions, replace with your variable if different
        device = self.device

        # Split and normalize embeddings
        embeddings = self.Gu.weight.view(N, M, self.embed_k)  # Shape: (N, M, embed_k)
        norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # Normalize along the embedding dimension

        loss = 0.0
        count = 0

        # Compute mutual information loss between each pair of embeddings
        for i in range(M):
            for j in range(M):
                if i != j:
                    # Compute similarities between embeddings[:, i, :] and embeddings[:, j, :]
                    similarities = torch.matmul(norm_embeddings[:, i, :], norm_embeddings[:, j, :].T)  # Shape: (N, N)
                    similarities /= temperature

                    # Labels for InfoNCE (diagonal elements are positives)
                    labels = torch.arange(N).to(device)

                    # Cross-entropy loss
                    loss += F.cross_entropy(similarities, labels)
                    count += 1

        # Average the loss over all pairs
        loss = loss / count
        return loss



    def train_step(self, batch):
        user, pos, neg = batch

        # # Positive items
        # xui_pos, gu_pos, gamma_i_pos = self.forward(inputs=(user[:,0], pos[:, 0]))
        # # Negative items
        # xui_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

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


        # ------------------ Rule Indipendence
        loss += self.l_rc * self.dissimilarity_loss_matrix()

        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
