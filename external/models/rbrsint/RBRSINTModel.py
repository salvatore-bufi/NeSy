from abc import ABC
import torch
import numpy as np
import random
import torch.nn.functional as F


class RBRSINTModel(torch.nn.Module, ABC):
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
        # User Embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gu.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)

        # Initialize Rules Embeddings
        self.Gr = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty((self.embed_k, self.n_rules ))))
        self.Gr.to(self.device)

        # Item embeddings
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gi.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def disjunction_rule(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        '''
        in the actual implementation, as the rules are learned per user, the selector is always an array of ones, as the scores (selected)
        are already done wrt the specific user-rule embedding
        * This function performs a smooth OR-like operation on two tensors: selector and selected. The selector tensor corresponds to binary values that determine which parts of the selected tensor are taken into account. The function computes the log of 1 - selector * selected (with a small epsilon added to prevent log(0)), sums the logs across the dimensions, and returns a final score between 0 and 1. This score represents the result of a differentiable OR operation, where higher values indicate stronger disjunction (OR).'''
        expr = torch.log(1 - selector * selected + epsilon)
        log_sum = torch.sum(expr, dim=1)
        return 1 - (-1 / (-1 + log_sum))

    def conjunction_rule(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        """
        this function can be used in some way, actually it is not used
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

        # Compute rules weight wrt user - rules weight
        scores = torch.matmul(gu, self.Gr)
        rules_weight = torch.softmax(scores, dim=1) # shape (batch_size, n_rules)


        # Expanding Torch tensor, for broadcasting
        # Compute user view enhanced through rules - u =  u  + rules_weight * rule
        # gu_expanded[0] = user_0 (n_rules, embed_k), initially it is repeated equal
        gu_expanded = gu.unsqueeze(1).expand(-1, self.n_rules, -1) # shape (n_user, n_rules, embed_k).

        # gr_expanded contains Gr repeated for batch_size, i.e. gr_exp[0].shape = n_rules, embed_k that is equal to gr_exp[1] etc.
        gr_expanded = self.Gr.T.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, n_rules, embed_k ]

        rules_weight_expanded = rules_weight.unsqueeze(2) # [batch_size, n_rules, 1]

        # Compute user-rule embedding:
        gu_r = rules_weight_expanded * gr_expanded + gu_expanded  # (n_user, n_rules, embed_k).

        # gi_expanded
        gi = self.Gi.weight[items]
        gi_expanded = gi.unsqueeze(1)

        # MF scores ( for each rule)
        scores = torch.sigmoid(torch.sum(gu_r * gi_expanded, dim=2)) # [batch_size, n_rules] | score[i] = score of user-item in batch for each rule
        final_score = self.disjunction_rule(selector=1, selected=scores) # [batch_size]
        return final_score, gu, gi


    def predict(self, start, stop, **kwargs):

        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)  # Shape: [num_users, n_rules * embed_k]
        batch_size = gu.shape[0]
        # Compute rules weight wrt user - rules weight
        scores = torch.matmul(gu, self.Gr)
        rules_weight = torch.softmax(scores, dim=1)  # shape (batch_size, n_rules)

        gu_expanded = gu.unsqueeze(1).expand(-1, self.n_rules, -1)  # shape (n_user, n_rules, embed_k).
        # gr_expanded contains Gr repeated for batch_size, i.e. gr_exp[0].shape = n_rules, embed_k that is equal to gr_exp[1] etc.
        gr_expanded = self.Gr.T.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_rules, embed_k ]

        rules_weight_expanded = rules_weight.unsqueeze(2)  # [batch_size, n_rules, 1]

        # Compute user-rule embedding:
        gu_r = rules_weight_expanded * gr_expanded + gu_expanded  # (n_user, n_rules, embed_k).

        ###### ------------------------------------------------------------------------------------------------

        gamma_i = self.Gi.weight.to(self.device)  # Shape: [num_items, embed_k]

        # Compute scores for each user and item
        scores = []
        for r in range(self.n_rules):
            gu_ri = gu_r[:, r, :]  # [num_users, embed_k]
            # Broadcasting over items
            and_score = torch.matmul(gu_ri, torch.transpose(gamma_i, 0, 1))  # [num_users, num_items]
            scores.append(and_score.unsqueeze(0))  # [1, num_users, num_items]

        # Combine scores using disjunction
        and_scores_tensor = torch.nn.functional.sigmoid(torch.cat(scores, dim=0))  # [n_rules, num_users, num_items]
        expr = 1 -  and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]

        return final_scores

    def diversity_loss(self):
        Gr = self.Gr.T
        # Normalize the rows to have unit length
        Gr_norm = F.normalize(Gr, p=2, dim=1)

        # Compute the cosine similarity matrix
        similarity_matrix = Gr_norm @ Gr_norm.t()

        # Exclude the diagonal elements (self-similarity)
        batch_size = Gr.size(0)
        mask = torch.eye(batch_size, device=Gr.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)

        # Compute the average of the off-diagonal similarities
        loss = similarity_matrix.sum() / (batch_size * (batch_size - 1))

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
        loss += self.l_rc * self.diversity_loss()

        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
