import torch
import numpy as np
import random
from abc import ABC

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

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.logic_w = 0.05

        # Embeddings for users and items
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k, device=self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k, device=self.device)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        torch.nn.init.xavier_uniform_(self.Gi.weight)

        # Convert ui to tensor on device
        self.ui = torch.tensor(ui, device=self.device, requires_grad=False)
        self.user_history = {int(u): self.ui[:, self.ui[0, :] == u][1, :] for u in self.ui[0, :].unique()}

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def like_history(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(u, i.T)).unsqueeze(0)

    def antecedent(self, x: torch.Tensor) -> torch.Tensor:
        return 1. - x

    def disjunction_nips(self, selector: torch.Tensor, selected: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
        expr = torch.log(1. - selector * selected + epsilon)
        log_sum = torch.sum(expr, dim=1)
        return 1. - (-1.0 / (-1.0 + log_sum))

    def disjunction_godel(self, selected: torch.Tensor) -> torch.Tensor:
        return torch.max(selected)



    def disjunction(self, selected: torch.Tensor, f = '2') -> torch.Tensor:
        if f == '1':
            return self.disjunction_godel(selected)
        else:
            return self.disjunction_nips(1.0, selected)



    def premise_user_history(self, u: int) -> torch.Tensor:
        user_history = self.user_history[u]
        gu = self.Gu.weight[u]
        gi = self.Gi(user_history)
        rule_antecedent = self.antecedent(self.like_history(gu, gi))
        # return self.disjunction(selector=1.0, selected=rule_antecedent)
        # return self.disjunction(rule_antecedent)
        return self.disjunction(rule_antecedent, f='1')
    def calculate_batch_premise(self, users: np.array):
        u_premis_list = [self.premise_user_history(u).unsqueeze(0) for u in users]
        return torch.cat(u_premis_list, dim=0)

    def forward_logic(self, inputs, **kwargs):
        users_premises, scores = inputs
        users_premises = users_premises.unsqueeze(1)
        ui = torch.sigmoid(scores).unsqueeze(1)
        # xui = self.disjunction(selector=1.0, selected=torch.cat((ui, users_premises), dim=1))
        xui = self.disjunction(torch.cat((ui, users_premises), dim=1), f='1')
        return xui

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


        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))

        premise = self.calculate_batch_premise(user[:, 0])
        logic_score_pos = self.forward_logic(inputs=(premise, xu_pos))
        logic_score_neg = self.forward_logic(inputs=(premise, xu_neg))
        logic_loss = -torch.mean(torch.nn.functional.logsigmoid(logic_score_pos - logic_score_neg))
        # print(f"\nBPR LOSS: {loss} \t LOGIC LOSS: {logic_loss}")
        loss += self.logic_w * logic_loss

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
