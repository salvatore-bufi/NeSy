from abc import ABC
import torch
import numpy as np
import random
import torch.nn.functional as F


class RBRSINT2Model(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 lr: float,
                 embed_k: int,
                 l_w: int,
                 random_seed: int,
                 n_rules: int,
                 nint:int,
                 epsilon: float,
                 l_rc: float,
                 name="RBRSINT2",
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
        self.lr = lr
        self.l_w = l_w
        self.n_rules = n_rules  # Number of conjunction rules
        self.epsilon = epsilon  # Small constant to prevent log(0)
        self.l_rc = l_rc
        self.nint = nint

        # Initialize embeddings for users and items
        # For users, we have n_rules different embeddings (one for each rule)
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gu.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)

        # Item embeddings
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        # torch.nn.init.trunc_normal_(self.Gi.weight, mean=0.5, std=1.0, a=0.0, b=1.0)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # Rules Embeddings
        self.Gr = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty((self.embed_k, self.nint * self.n_rules))))
        self.Gr.to(self.device)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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

        # Get user embeddings and reshape for rules
        gu = self.Gu.weight[users]

        # u_emb = []
        # for r in range(self.n_rules):
        #     u_int_r = torch.softmax(gu @ self.Gr[:, r*self.nint: (r+1)*self.nint]) @ self.Gr[:, r*self.nint: (r+1)*self.nint].T
        #     u_emb += u_int_r
        # u_emb = torch.stack([gu + x for x in u_emb], dim=1)

        # Get item embeddings
        gamma_i = self.Gi.weight[items]  # Shape: [batch_size, embed_k]

        # Compute the score of each rule using MF
        and_scores = []
        for r in range(self.n_rules):
            u_int_r = torch.softmax(gu @ self.Gr[:, r * self.nint: (r + 1) * self.nint], dim=1) @ self.Gr[:, r * self.nint: (r + 1) * self.nint].T
            gamma_u_r = gu + u_int_r
            score_and = torch.sum(gamma_u_r * gamma_i, -1)
            and_scores.append(score_and.unsqueeze(1))  # Shape: [batch_size, 1]

        # Concatenate and compute disjunction
        and_scores_tensor = torch.sigmoid(torch.cat(and_scores, dim=1))  # Shape: [batch_size, n_rules]
        xui = self.disjunction_rule(1, and_scores_tensor)  # Shape: [batch_size]

        return xui, gu, gamma_i

    def predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        gu = self.Gu.weight[start:stop].to(self.device)  # Shape: [num_users, n_rules * embed_k]

        # selector = torch.ones(self.num_users, self.num_items)
        selector = 1
        gamma_i = self.Gi.weight.to(self.device)  # Shape: [num_items, embed_k]

        # Compute scores for each user and item
        scores = []
        for r in range(self.n_rules):
            u_int_r = torch.softmax(gu @ self.Gr[:, r * self.nint: (r + 1) * self.nint], dim=1) @ self.Gr[:, r * self.nint: (                                                                                                                   r + 1) * self.nint].T
            gu_r = gu + u_int_r
            # Broadcasting over items
            and_score = torch.matmul(gu_r, torch.transpose(gamma_i, 0, 1))  # [num_users, num_items]
            scores.append(and_score.unsqueeze(0))  # [1, num_users, num_items]

        # Combine scores using disjunction
        and_scores_tensor = torch.sigmoid(torch.cat(scores, dim=0))  # [n_rules, num_users, num_items]
        expr = 1 - selector * and_scores_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]

        return final_scores




    def mutual_information_loss(self, temperature=0.2):
        rules = self.Gr.T
        embeddings = rules.view(self.nint, self.n_rules, self.embed_k)
        norm_embeddings = F.normalize(embeddings, p=2, dim=2)
        N, num_rules, M = embeddings.shape
        # Reshape to [N * num_rules, M] for InfoNCE
        reshaped_embeddings = embeddings.view(N * num_rules, M)  # [N*num_rules, M]
        # Compute similarity matrix
        similarity_matrix = torch.matmul(reshaped_embeddings, reshaped_embeddings.T)  # [N*num_rules, N*num_rules]
        similarity_matrix /= temperature

        # Labels: for within-row, positives are embeddings of the same sample but different rules
        # This is more complex, but one approach is to treat each embedding as a query and its positives as other embeddings from the same sample.

        # However, InfoNCE typically expects one positive per query. To adapt, we can use multiple positives or use a different loss like contrastive loss.

        # For simplicity, we'll assume each embedding's positive is its corresponding embedding in another rule for the same sample.
        # For example, for each sample, rule 0's positive could be rule 1's embedding, etc.
        # This requires defining positive pairs appropriately.

        # Alternatively, to keep it simple, treat each embedding as its own class in the combined loss.
        # This approach encourages all embeddings to be distinct, but might not directly capture within-row relationships.

        # Here, we'll proceed with treating each embedding as its own class.
        labels = torch.arange(N * num_rules).to(self.device)

        # For numerical stability, subtract the max from each row
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True).values

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

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
        reg_loss = self.l_w * (1/2) * (
                gu_pos.norm(2).pow(2) +
                gamma_i_pos.norm(2).pow(2) +
                gamma_i_neg.norm(2).pow(2)
        ) / user.shape[0]

        loss += reg_loss

        reg_rules_loss = self.l_w * (self.Gr.norm(2).pow(2)) / (self.nint )
        loss += reg_rules_loss
        # ------------------ Rule Indipendence
        loss += self.l_rc * self.mutual_information_loss()



        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
