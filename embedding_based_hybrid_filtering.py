import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Example embeddings for artworks (replace with actual data)
np.random.seed(42)  # For reproducibility
num_artworks = 150  # Total number of artworks
embedding_dim = 128  # Dimensionality of embeddings
all_embeddings = np.random.rand(num_artworks, embedding_dim)

# Create interaction matrix
user_interactions = [
    [0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [3, 100], [4, 100],
    [3, 101], [4, 101], [3, 102]
]
user_ids = [interaction[0] for interaction in user_interactions]
data_point_indices = [interaction[1] for interaction in user_interactions]

n_users = max(user_ids) + 1
n_items = all_embeddings.shape[0]  # Use number of embeddings as items
data = [1] * len(user_interactions)
interaction_matrix = csr_matrix((data, (user_ids, data_point_indices)), shape=(n_users, n_items))

# Compute user-user similarity
user_similarity = cosine_similarity(interaction_matrix)

# Combine embeddings for hybrid recommendation
def hybrid_embedding_user_user(user_id, all_embeddings, interaction_matrix, user_similarity, alpha=0.5, top_k=5):
    """
    Combines embedding-based and user-user collaborative filtering for hybrid recommendation.
    """
    # Get user-specific similarity
    similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude the user
    target_user_items = set(interaction_matrix[user_id].nonzero()[1])

    # Initialize scores
    item_scores = np.zeros(all_embeddings.shape[0])

    # User-user contribution
    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]
        similar_user_items = interaction_matrix[similar_user].nonzero()[1]

        for item in similar_user_items:
            item_scores[item] += similarity_score

    # Embedding contribution
    target_user_embedding = np.mean(
        all_embeddings[list(target_user_items)], axis=0
    ) if target_user_items else np.zeros(all_embeddings.shape[1])
    embedding_scores = cosine_similarity([target_user_embedding], all_embeddings)[0]

    # Combine the scores
    combined_scores = alpha * item_scores + (1 - alpha) * embedding_scores
    recommended_items = np.argsort(-combined_scores)[:top_k]

    return recommended_items

# # Example usage
# user_id = 2
# recommended_items = hybrid_embedding_user_user(user_id, all_embeddings, interaction_matrix, user_similarity, alpha=0.7, top_k=5)
# print("Hybrid Model 1 Recommendations:", recommended_items)


def hybrid_weighted_recommendation(user_id, interaction_matrix, user_similarity, all_embeddings, alpha=0.5, top_k=5):
    """
    Combines user-user collaborative filtering with content-based filtering using weighted scores.
    """
    # User-user collaborative filtering
    similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude the user
    target_user_items = set(interaction_matrix[user_id].nonzero()[1])
    item_scores_cf = {}

    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]
        similar_user_items = interaction_matrix[similar_user].nonzero()[1]

        for item in similar_user_items:
            if item not in target_user_items:  # Exclude items already interacted with
                item_scores_cf[item] = item_scores_cf.get(item, 0) + similarity_score

    # Normalize collaborative filtering scores
    max_cf_score = max(item_scores_cf.values()) if item_scores_cf else 1
    for item in item_scores_cf:
        item_scores_cf[item] /= max_cf_score

    # Content-based filtering
    target_user_embedding = np.mean(
        all_embeddings[list(target_user_items)], axis=0
    ) if target_user_items else np.zeros(all_embeddings.shape[1])
    embedding_scores = cosine_similarity([target_user_embedding], all_embeddings)[0]

    # Combine scores
    final_scores = np.zeros(all_embeddings.shape[0])
    for item, score in item_scores_cf.items():
        final_scores[item] += alpha * score
    final_scores += (1 - alpha) * embedding_scores

    # Sort and recommend top-k items
    recommended_items = np.argsort(-final_scores)[:top_k]
    return recommended_items

# Example usage
user_id = 2
recommended_items = hybrid_weighted_recommendation(user_id, interaction_matrix, user_similarity, all_embeddings, alpha=0.6, top_k=5)
print("Hybrid Model 2 Recommendations:", recommended_items)

