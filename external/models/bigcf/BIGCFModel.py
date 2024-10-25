
from abc import ABC

import torch_sparse
from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

class BIGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 nint: int,
                 n_layers: int,
                 embed_k: int,
                 learning_rate: int,
                 l_2: float,
                 l_c: float,
                 k: int,
                 t: int,
                 adj: torch_sparse.SparseTensor,
                 random_seed: int,
                 name="BIGCF",
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

        #------------------ Embedding and parameters initialization

        #------- User and Item Embedding: U = (num_user, dim_embedding) ; I = (num_item, dim_embedding)
        self.user_emb = nn.Embedding(self.num_users, self.embed_k)
        self.item_emb = nn.Embedding(self.num_items, self.embed_k)

        #------- User and Item Intent initialization
        # user_intent =  (dim_embedding, num_intent)
        self.user_intent = nn.Parameter(
            nn.init.xavier_normal_(torch.empty((self.embed_k, self.nint))), requires_grad=True
        )
        # item_intent =  (dim_embedding, num_intent)
        self.item_intent = nn.Parameter(
            nn.init.xavier_normal_(torch.empty((self.embed_k, self.nint))), requires_grad=True
        )

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
        # all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        # ----------------- Bilateral Intent
        u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
        i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T

        int_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

        #----------------- Reparameterization
        noise = torch.randn_like(all_embeddings)
        all_embeddings = all_embeddings + int_embeddings * noise

        # ---------------- Final emb
        user_embedding, item_embedding = torch.split(all_embeddings, [self.num_users, self.num_items], 0)
        return user_embedding, item_embedding, all_embeddings, int_embeddings


    def forward(self, inputs, **kwargs):
        '''
                gu = u-emb: batch x dim_emb
                gi = i-emb: batch x dim_emb
                return score: batch_size * 1 (each element = score u-i)
                '''
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)
        xui = torch.sum(gamma_u * gamma_i, 1)
        return xui
    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb):
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.t)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.t), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        u_gnn_embs, i_gnn_embs = torch.split(gnn_emb, [self.num_users, self.num_items], 0)
        u_int_embs, i_int_embs = torch.split(int_emb, [self.num_users, self.num_items], 0)

        u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
        u_int_embs = F.normalize(u_int_embs[users], dim=1)

        i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
        i_int_embs = F.normalize(i_int_embs[items], dim=1)

        cl_loss += cal_loss(u_gnn_embs, u_gnn_embs)
        cl_loss += cal_loss(i_gnn_embs, i_gnn_embs)
        cl_loss += cal_loss(u_gnn_embs, i_gnn_embs)

        cl_loss += cal_loss(u_int_embs, u_int_embs)
        cl_loss += cal_loss(i_int_embs, i_int_embs)
        return cl_loss

    def train_step(self, batch):

        # retrieve  embeddings
        # int_emb = only intent embeddings
        # full_emb = final embeddings ( i.e., gu, gi after intent )
        gu, gi, full_emb, int_emb = self.propagate_embeddings()

        # user_ids, positive_items_ids, negative_items_ids
        user_t, pos_t, neg_t = batch
        user, pos, neg = user_t[:, 0], pos_t[:, 0], neg_t[:, 0]

        # ELBO Loss
        xu_pos = self.forward(inputs=(gu[user], gi[pos]))  # positive score
        xu_neg = self.forward(inputs=(gu[user], gi[neg]))  # negative score
        # difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        # loss = torch.sum(self.softplus(-difference))
        loss = torch.mean(F.softplus(xu_neg - xu_pos))

        # L2 Regularization Loss
        l2_reg_loss = self.l_2 * (self.user_emb.weight[user].norm(2).pow(2) +
                               self.item_emb.weight[pos].norm(2).pow(2) +
                               self.item_emb.weight[neg].norm(2).pow(2))
        loss += l2_reg_loss

        # L2 Intent Reg. Loss
        l2_intent_reg_loss = self.l_2 * (self.user_intent.norm(2).pow(2) +
                                      self.item_intent.norm(2).pow(2))
        loss += l2_intent_reg_loss

        # graph contrastive loss
        gcl_loss = self.l_c * self.cal_ssl_loss(user, pos, full_emb, int_emb)
        loss += gcl_loss



        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
