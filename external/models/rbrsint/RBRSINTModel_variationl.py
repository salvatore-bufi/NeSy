import torch
import torch.nn.functional as F
import numpy as np
import random
from abc import ABC

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
                 name="RBRSINT",
                 **kwargs):
        super().__init__()

        # Set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.n_rules = n_rules
        self.epsilon = epsilon
        self.l_rc = l_rc

        # Encoder parameters: Mean and log-variance of user embeddings
        self.Gu_mean = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu_mean.weight)
        self.Gu_mean.to(self.device)

        # Rule embeddings
        self.Gr = torch.nn.Parameter(torch.Tensor(self.n_rules, self.embed_k))
        torch.nn.init.xavier_uniform_(self.Gr)
        self.Gr.to(self.device)

        # Item embeddings
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, inputs, **kwargs):
        users, items = inputs
        batch_size = users.shape[0]

        # Get mean and log-variance of user embeddings
        gu_mean = self.Gu_mean.weight[users] # [batch_size, embed_k]


        # Rule embeddings
        gr = self.Gr  # [n_rules, embed_k]

        # Compute attention scores s_{jk} between Gumean and Gr[k]
        scores = torch.matmul(gu_mean, torch.exp(0.5 * gr).T)  # [batch_size, n_rules]
        s_jk = torch.softmax(scores, dim=1)  # [batch_size, n_rules]

        # Expand dimensions for computation
        gu_mean_expanded = gu_mean.unsqueeze(1).expand(-1, self.n_rules, -1)  # [batch_size, n_rules, embed_k]
        gr_expanded = gr.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_rules, embed_k]
        s_jk_expanded = s_jk.unsqueeze(2)  # [batch_size, n_rules, 1]

        # Compute user-rule embeddings: e_{ujk} = z + s_{jk} * Gr[k]
        gu_logvar = s_jk_expanded * gr_expanded
        e_ujk = self.reparameterize(mu=gu_mean_expanded, logvar=gu_logvar)  # [batch_size, n_rules, embed_k]

        # Get item embeddings
        gamma_i = self.Gi.weight[items]  # [batch_size, embed_k]

        # Compute the score of each rule using MF
        and_scores = torch.sum(e_ujk * gamma_i.unsqueeze(1), dim=2)  # [batch_size, n_rules]

        # TODO: ERRORE NELL OR
        # Apply sigmoid to get probabilities
        and_scores_tensor = torch.sigmoid(and_scores)  # [batch_size, n_rules]

        # Compute disjunction (smooth OR)
        xui = 1 - torch.prod(1 - and_scores_tensor + self.epsilon, dim=1)  # [batch_size]

        # Also return KL divergence components
        return xui, gu_mean, gu_logvar

    def compute_kl_loss(self, mu, logvar):
        """
        Computes the KL divergence between the approximate posterior and the prior.
        """
        # Prior is N(0, I)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_divergence.mean()

    def compute_kl_loss_over_2_dimension(self, mu, logvar):
        """
        Computes the KL divergence between the approximate posterior and the prior.
        """
        # Prior is N(0, I)
        kl_divergence  = 0
        for i in range(self.n_rules):
            kl_divergence += -0.5 * torch.sum(1 + logvar[:, i, :] - mu.pow(2) - logvar[:, i, :].exp(), dim=1)
        return kl_divergence.mean()

    def predict(self, start, stop, **kwargs):
        # Prediction for all items for users in the range [start, stop)
        with torch.no_grad():
            users = torch.arange(start, stop).to(self.device)

            # Use the mean of the user embeddings for prediction
            gu_mean = self.Gu_mean(users)  # [num_users, embed_k]

            # Rule embeddings
            gr = self.Gr  # [n_rules, embed_k]

            # Compute attention scores s_{jk}
            scores = torch.matmul(gu_mean, torch.exp(0.5 * gr).T)  # [num_users, n_rules]
            s_jk = torch.softmax(scores, dim=1)  # [num_users, n_rules]

            # Expand dimensions
            gu_mean_expanded = gu_mean.unsqueeze(1).expand(-1, self.n_rules, -1)  # [num_users, n_rules, embed_k]
            gr_expanded = gr.unsqueeze(0).expand(gu_mean.size(0), -1, -1)  # [num_users, n_rules, embed_k]
            s_jk_expanded = s_jk.unsqueeze(2)  # [num_users, n_rules, 1]

            # Compute e_ujk
            gu_logvar = s_jk_expanded * gr_expanded
            e_ujk = self.reparameterize(mu=gu_mean_expanded, logvar=gu_logvar)  # [num_users, n_rules, embed_k]

            # Item embeddings
            gamma_i = self.Gi.weight.to(self.device)  # [num_items, embed_k]

            # Compute scores
            scores = torch.matmul(e_ujk, gamma_i.T)  # [num_users, n_rules, num_items]

            # Apply sigmoid
            scores = torch.sigmoid(scores)  # [num_users, n_rules, num_items]

            # Compute disjunction
            expr = 1 - scores + self.epsilon
            final_scores = 1 - torch.prod(expr, dim=1)  # [num_users, num_items]

            return final_scores

    def train_step(self, batch):
        user, pos, neg = batch

        # Positive items
        xui_pos, gu_mean_pos, gu_logvar_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        # Negative items
        xui_neg, gu_mean_neg, gu_logvar_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        # BPR Loss
        diff_scores = xui_pos - xui_neg  # [batch_size]
        bpr_loss = torch.nn.functional.softplus(-diff_scores).mean()

        # KL Divergence Loss
        kl_loss = self.compute_kl_loss_over_2_dimension(gu_mean_pos, gu_logvar_pos)

        # Regularization Loss
        reg_loss = self.l_w * (
            self.Gu_mean.weight.norm(2).pow(2) +
            self.Gi.weight.norm(2).pow(2) +
            self.Gr.norm(2).pow(2)
        ) / user.shape[0]

        # Total Loss
        loss = bpr_loss + kl_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)









