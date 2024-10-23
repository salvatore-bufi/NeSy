from abc import ABC

import torch_sparse
from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

class RBRSGNNMULTIPLEModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 nint: int,
                 nr: int,
                 n_layers: int,
                 embed_k: int,
                 learning_rate: int,
                 l_2: float,
                 l_c: float,
                 k: int,
                 t: int,
                 adj: torch_sparse.SparseTensor,
                 random_seed: int,
                 name="RBRSGNNMULTIPLE",
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
        self.nint = nint
        self.n_layers = n_layers
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_2 = l_2
        self.l_c = l_c
        self.k = k
        self.t = t
        self.adj = adj
        self.normalize = True
        self.nr = nr
        self.epsilon = 1e-40

        #------------------ Embedding and parameters initialization

        #------- User and Item Embedding: U = (num_user, dim_embedding) ; I = (num_item, dim_embedding)
        self.user_emb = nn.Embedding(self.num_users, self.embed_k)
        self.item_emb = nn.Embedding(self.num_items, self.embed_k)

        #------- User Rules initialization
        # user_intent =  (dim_embedding, num_intent)
        self.user_intent = nn.Parameter(
            nn.init.xavier_normal_(torch.empty((self.embed_k, self.nint * self.nr))), requires_grad=True
        ) #nint = number of int per rule, nr= number of rules


        # ------- GNN  initialization
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(normalize=self.normalize), 'x, edge_index -> x'))
        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        # ------- Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.user_emb.weight.to(self.device), self.item_emb.weight.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

    # GNN propagation
        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        #----------------- User and Item GNN embeddings
        all_embeddings = torch.mean(torch.stack(all_embeddings, 0), dim=0)
        u_embeddings, item_embedding = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        # -----------------  Intent
        u_int_embeddings_1 = torch.softmax(u_embeddings @ self.user_intent[:, :int((self.nr/2))*self.nint], dim=1) @ self.user_intent[:, :int((self.nr/2))*self.nint].T
        u_int_embeddings_2 = torch.softmax(u_embeddings @ self.user_intent[:, int((self.nr/2))*self.nint:], dim=1) @ self.user_intent[:, int((self.nr/2))*self.nint:].T

        # int_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)
        u_int_embeddings_1 += u_embeddings
        u_int_embeddings_2 += u_embeddings
        # ---------------- Final emb
        user_embedding = torch.cat((u_int_embeddings_1, u_int_embeddings_2), dim=1)
        return user_embedding, item_embedding

    def disjunction_rule(self, rules_score: torch.Tensor) -> torch.Tensor:
        # Compute the disjunction (OR) function
        # rules_score is a tensor of shape [batch_size, n_rules], i.e. rules_score.shape[0] == batch_size, rules_score.shape[1] == n_rules
        expr = 1 - rules_score + self.epsilon
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=1)
        res = 1 - (-1.0 / (-1.0 + sum_log_expr))
        return res

    def forward(self, inputs, **kwargs):
        '''
                gu = u-emb: batch x dim_emb
                gi = i-emb: batch x dim_emb
                return score: batch_size * 1 (each element = score u-i)
                '''
        gu, gi = inputs
        gu = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        predicate_scores = []
        for r in range(self.nr):
            gamma_u_r = gu[:, r*self.embed_k : (r+1)* self.embed_k]  # extract the respective intent
            predicate_score = torch.sum(gamma_u_r * gamma_i, -1)
            predicate_scores.append(predicate_score.unsqueeze(1)) # Shape: [batch_size, 1]

        predicate_score_tensor = torch.nn.functional.sigmoid(torch.cat(predicate_scores, dim=1))  # Shape: [batch_size, n_rules]
        xui = self.disjunction_rule(predicate_score_tensor)  # Shape: [batch_size]
        return xui
    def predict(self, gu, gi, **kwargs):
        predicate_scores = []
        for r in range(self.nr):
            gu_r = gu[:, r*self.embed_k : (r+1)* self.embed_k]
            predicate_score = torch.matmul(gu_r, torch.transpose(gi, 0, 1)) # Shape: [num_users, num_items]
            predicate_scores.append(predicate_score.unsqueeze(0)) # Shape: [1, num_users, num_items]

        # Combine scores using disjunction
        predicate_score_tensor = torch.nn.functional.sigmoid(torch.cat(predicate_scores, dim=0)) # n_rules, num_users, num_items
        expr = 1 - predicate_score_tensor + self.epsilon  # [n_rules, num_users, num_items]
        log_expr = torch.log(expr)
        sum_log_expr = torch.sum(log_expr, dim=0)  # Sum over rules
        final_scores = 1 - (-1.0 / (-1.0 + sum_log_expr))  # [num_users, num_items]
        return final_scores


    def dissimilarity_loss(self,gu, margin=0.5):
        r0, r1 = torch.split(gu, [self.embed_k, self.embed_k], 1)
        r0_normalized = F.normalize(r0, p=2, dim=1)
        r1_normalized = F.normalize(r1, p=2, dim=1)
        similarities = torch.sum(r0_normalized * r1_normalized, dim=1)
        loss = F.relu(similarities + margin).mean()
        return loss

    def mutual_information_loss(self, gu, temperature=0.2):
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
        r0, r1 = torch.split(gu, [self.embed_k, self.embed_k], 1)
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

    def train_step(self, batch):

        # retrieve  embeddings
        # int_emb = only intent embeddings
        # full_emb = final embeddings ( i.e., gu, gi after intent )
        gu, gi= self.propagate_embeddings()

        # user_ids, positive_items_ids, negative_items_ids
        user_t, pos_t, neg_t = batch
        user, pos, neg = user_t[:, 0], pos_t[:, 0], neg_t[:, 0]

        # ELBO Loss
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))  # positive score
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))  # negative score
        # difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        # loss = torch.sum(self.softplus(-difference))
        loss = torch.sum(F.softplus(xu_neg - xu_pos))

        # L2 Regularization Loss
        l2_reg_loss = self.l_2 * (self.user_emb.weight[user].norm(2).pow(2) +
                               self.item_emb.weight[pos].norm(2).pow(2) +
                               self.item_emb.weight[neg].norm(2).pow(2)) / gu[user].shape[0]
        l2_intent_loss = self.l_2 * (self.user_intent.norm(2).pow(2)) / self.nint

        # print(f"\n BPR Loss: {loss} \t \t L2 Loss: {l2_reg_loss}")
        loss += l2_reg_loss
        loss += l2_intent_loss

        # --- Rule independence loss
        rule_ind = self.l_c * self.mutual_information_loss(gu[user])
        loss += rule_ind


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)