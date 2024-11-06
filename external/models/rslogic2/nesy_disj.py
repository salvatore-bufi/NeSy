import torch

'''Use Softmax-based disjunction if you want the most relevant embeddings to have the strongest influence.
Use Max-pooling if you only want the top preference to determine the score.
Use Log-sum-exp if you need smooth differentiation with tunable sensitivity.
Use Product-based disjunction if embeddings are normalized to [0, 1] and you want cumulative influence.'''
class NeuroSymbolicDisjunction:
    def __init__(self, method="log_sum_exp", alpha=1.0, epsilon: float=1e-40):
        self.method = method
        self.alpha = alpha  # Controls smoothness for log-sum-exp
        self.epsilon = epsilon

    def forward(self, x):
        if self.method == "softmax":
            return torch.sum(torch.softmax(x, dim=1) * x, dim=1)

        elif self.method == "max_pooling":
            return torch.max(x, dim=1)[0]

        elif self.method == "log_sum_exp":
            return (1 / self.alpha) * torch.logsumexp(self.alpha * x, dim=1)

        elif self.method == "product_prob":
            return 1 - torch.prod(1 - x, dim=1)

        # method implemented in Scalable Rule-Based Representation Learning for Interpretable Classification
        elif self.method == 'rrl':
            expr = torch.log(1. - x + self.epsilon)
            log_sum = torch.sum(expr, dim=1)
            return 1. - (-1.0 / ( -1.0 + log_sum))

        else:
            raise ValueError(f"Unknown method: {self.method}")

'''class RecommenderWithDisjunction(torch.nn.Module):
    def __init__(self, num_users, num_items, embed_dim, disjunction_method="log_sum_exp"):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim)
        self.disjunction = NeuroSymbolicDisjunction(method=disjunction_method)

    def forward(self, user_ids, item_ids):
        # Get user and item embeddings
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Compute similarity scores
        interaction_scores = torch.sigmoid(torch.matmul(user_embed, item_embed.T))

        # Apply disjunction over item scores for each user
        final_scores = self.disjunction.forward(interaction_scores)

        return final_scores
        '''