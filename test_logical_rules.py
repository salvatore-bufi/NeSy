import torch
import torch.nn.functional as F



def disjunction(selector: torch.Tensor, selected:torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
    expr = torch.log(1 - selector * selected + epsilon)
    log_sum = torch.sum( expr, dim=1)
    return 1 - (-1 / (-1 + log_sum))


def conjunction_rule(u: torch.Tensor, i: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
    """
    Compute the conjunction (AND) function.

    Args:
        u (torch.Tensor): Tensor of shape [batch_size, embed_k], representing user embeddings.
        i (torch.Tensor): Tensor of shape [batch_size, embed_k], representing item embeddings.
        epsilon (float, optional): Small constant to prevent log(0). Defaults to 1e-12.

    Returns:
        torch.Tensor: Tensor of shape [batch_size], containing the conjunction scores.
    """
    # Ensure u and i are in the expected range [0, 1]
    # u = torch.clamp(u, 0.0, 1.0)
    # i = torch.clamp(i, 0.0, 1.0)

    # Compute the element-wise terms
    expr = 1 - u * (1 - i) + epsilon  # epsilon prevents log(0)

    # Take the logarithm of each term
    log_expr = torch.log(expr)

    # Sum over the embedding dimensions
    sum_log_expr = torch.sum(log_expr, dim=1)

    # Compute the final AND function value
    res = -1.0 / (-1.0 + sum_log_expr)
    return res


def disjunction_rule(and_rules: torch.Tensor, epsilon: float = 1e-40) -> torch.Tensor:
    """
    Compute the disjunction (OR) function.

    Args:
        and_rules (torch.Tensor): Tensor of shape [batch_size, n_rules], containing conjunction scores.
        epsilon (float, optional): Small constant to prevent log(0). Defaults to 1e-12.

    Returns:
        torch.Tensor: Tensor of shape [batch_size], containing the disjunction scores.
    """
    # Ensure and_rules are in the expected range [0, 1]
    # and_rules = torch.clamp(and_rules, 0.0, 1.0)

    # Compute the element-wise terms
    expr = 1 - and_rules + epsilon  # epsilon prevents log(0)

    # Take the logarithm of each term
    log_expr = torch.log(expr)

    # Sum over the rules
    sum_log_expr = torch.sum(log_expr, dim=1)

    # Compute the final OR function value
    res = 1 - (-1.0 / (-1.0 + sum_log_expr))
    return res


# users = torch.tensor([[1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1]])
# items = torch.tensor([[0, 1, 1, 0.2], [0, 0, 0, 1], [0.9, 1, 1, 1]])


users = torch.tensor([[0.9, 0, 0, 0.7], [0, 0, 0., 0.2], [0.9, 1, 1, 0.89]])
items = torch.tensor([[0, 0.7, 0.6, 0.2], [0, 0, 0, 1], [0.9, 1, 1, 0.97]])
# users = torch.tensor([[0.7, 0, 0, 0.9], [0, 0, -0.98, -1], [1, 1, 1, -1]])
# items = torch.tensor([[0, 1, 1, 0.2], [0, 0, 0.94, -1], [0.9, 1, 1, -0.97]])
print(f"{users} \n {items} \n Or Score {disjunction(users, items)} \n \n")

users = torch.tensor([[0.01, 0.2, 0.02, 0.02], [0, 0, 0., 0.], [0., 0, 0, 1]])
items = torch.ones(users.shape)
print(f"{users} \n {items} \n Or Score {disjunction(users, items)} \n \n")

res_ands = conjunction_rule(u=users, i=items)


# users = torch.tensor([[0.9, 0, 0, 0.7], [0, 0, 0., 0.2], [0.9, 1, 1, -0.89]])
# items = torch.tensor([[0, 0.7, 0.6, 0.2], [0, 0, 0, 1], [0.9, 1, 1, -0.97]])
users = torch.tensor([[0.7, 0, 0, 0.9], [0, 0, -0.98, -1], [1, 1, -1, -1]])
items = torch.tensor([[0, 1, 1, 0.1], [0, 0, 0.94, -1], [0.9, 1, 1, -0.97]])
res_ands = conjunction_rule(u=users, i=items)
print(res_ands)


ands = torch.tensor([[1, 0, 0, 0],
                     [0, 0.2, 0.3, 0.2],
                     [0, 0.95, 0.7, 0.4]])
print()

user_mul = torch.tensor([[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]])
item_mul = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]])
N, M = user_mul.size(0), item_mul.size(0)

def disjunction_rule_mf( and_rules: torch.Tensor) -> torch.Tensor:
    # Map unbounded inputs to probabilities in [0, 1]
    probabilities = torch.sigmoid(and_rules)
    # Compute the disjunction (OR) over these probabilities
    expr = 1 - probabilities + 1e-40
    log_expr = torch.log(expr)
    sum_log_expr = torch.sum(log_expr, dim=1)
    # Compute the final result
    res = 1 - torch.exp(sum_log_expr)
    return res
