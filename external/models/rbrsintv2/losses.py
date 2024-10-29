def dissimilarity_loss_matrix_over_gr(self, margin=0.5):
    rules = self.Gr.T

    # Split the embeddings into M parts
    embeddings = rules.view(self.nint, self.n_rules, self.embed_k)  # Shape: (N, M, embed_k)

    # Normalize embeddings
    norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # Normalize along the embedding dimension

    # Compute pairwise similarities between embeddings of different parts
    # similarities: Tensor of shape (N, M, M), where similarities[n, i, j] = similarity between
    # the i-th and j-th embeddings for the n-th sample
    similarities = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))  # Shape: (N, M, M)

    # Exclude self-similarities (diagonal elements) and duplicate pairs
    batch_size = embeddings.size(0)
    idx = torch.triu_indices(self.n_rules, self.n_rules, offset=1)
    pairwise_similarities = similarities[:, idx[0], idx[1]]  # Shape: (N, num_pairs)

    # Compute the loss
    loss = F.relu(pairwise_similarities + margin).mean()  # Mean over all pairs and samples
    return loss


def mutual_information_loss_vectorized_over_gr(self, temperature=0.2):
    """
    Minimizes mutual information between embeddings using InfoNCE loss.
    """
    rules = self.Gr.T
    # Split the embeddings into M parts
    embeddings = rules.view(self.nint, self.n_rules, self.embed_k)  # Shape: (N, M, embed_k)

    # Split and normalize embeddings
    norm_embeddings = F.normalize(embeddings, p=2, dim=2)  # Normalize along the embedding dimension

    loss = 0.0
    count = 0

    # Compute mutual information loss between each pair of embeddings
    for i in range(self.n_rules):
        for j in range(self.n_rules):
            if i != j:
                # Compute similarities between embeddings[:, i, :] and embeddings[:, j, :]
                similarities = torch.matmul(norm_embeddings[:, i, :], norm_embeddings[:, j, :].T)  # Shape: (N, N)
                similarities /= temperature

                # Labels for InfoNCE (diagonal elements are positives)
                labels = torch.arange(self.nint).to(self.device)

                # Cross-entropy loss
                loss += F.cross_entropy(similarities, labels)
                count += 1

    # Average the loss over all pairs
    loss = loss / count
    return loss