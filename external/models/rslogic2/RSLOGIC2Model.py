import torch
import numpy as np
import torch.nn.functional as F
from abc import ABC
import random

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

        # User and item embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k, device=self.device)
        torch.nn.init.xavier_uniform_(self.Gu.weight)

        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k, device=self.device)
        torch.nn.init.xavier_uniform_(self.Gi.weight)




        self.ui = torch.tensor(ui, device=self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def like_history(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # calculate the score of each u-i pair in the history
        return torch.sigmoid(torch.matmul(u.to(self.device), torch.transpose(i.to(self.device), 0, 1))).unsqueeze(0)

    def antecedent(self, x: torch.Tensor) -> torch.Tensor:
        # from implication to dnf form
        return 1. - x

    def disjunction(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        expr = torch.log(1. - selector * selected + epsilon)
        log_sum = torch.sum(expr, dim=1)
        # log_sum = torch.sum(expr)
        return 1. - (-1.0 / (-1.0 + log_sum))



    def premise_user_history(self, u: int) -> torch.Tensor:
        ## calculate the score of the part before implication (i.e. history)
        user_history = self.ui[:, self.ui[0, :] == u][1, :] ## torch tensor of lenght == number_of_items the user interacted
        gu = self.Gu.weight[u]
        gi = self.Gi.weight[user_history]
        rule_antecedent = self.antecedent(self.like_history(gu, gi))
        return self.disjunction(selector=1.0, selected=rule_antecedent)

    def calculate_batch_premise(self, users: np.array):
        ''' user: np.array of indices of users'''
        u_premis_list = []
        for u in users:
            premise = self.premise_user_history(u)
            u_premis_list.append(premise)
        return torch.cat(u_premis_list, dim=0)

    def forward_logic(self, inputs, **kwargs):
        users, items, scores = inputs
        users_premises = self.calculate_batch_premise(users).unsqueeze(1)
        ui = torch.sigmoid(scores).unsqueeze(1)
        xui = self.disjunction(selector=1.0, selected=torch.cat((ui, users_premises), dim=1))
        return xui

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop].to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        logic_score_pos = self.forward_logic(inputs=(user[:, 0], pos[:, 0], xu_pos))
        logic_score_neg = self.forward_logic(inputs=(user[:, 0], neg[:, 0], xu_neg))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))

        logic_loss = -torch.mean(torch.nn.functional.logsigmoid(logic_score_pos - logic_score_neg))
        loss += logic_loss

        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
