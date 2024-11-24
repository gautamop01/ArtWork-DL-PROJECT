import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Example embeddings for artworks (replace with your actual embeddings)
np.random.seed(42)  # For reproducibility
num_artworks = 150  # Total number of artworks
embedding_dim = 128  # Dimensionality of embeddings
all_embeddings = np.random.rand(num_artworks, embedding_dim)  # Randomly generated embeddings

# Create interaction matrix data
user_interactions = [
    [0, 1], [0, 2], [1, 1], [1, 2], [2, 1], [3, 100], [4, 100],
    [3, 101], [4, 101], [3, 102]
]
user_ids = [interaction[0] for interaction in user_interactions]
data_point_indices = [interaction[1] for interaction in user_interactions]

# Define the number of users and items
n_users = max(user_ids) + 1
n_items = all_embeddings.shape[0]  # Use the number of embeddings as the number of items

# Construct the interaction matrix
data = [1] * len(user_interactions)  # Interaction strength (binary: 1 for interaction, 0 otherwise)
interaction_matrix = csr_matrix((data, (user_ids, data_point_indices)), shape=(n_users, n_items))

# Compute user-user similarity matrix
user_similarity = cosine_similarity(interaction_matrix)

# Recommendation function
def recommend_items_user_user(user_id, interaction_matrix, user_similarity, top_k=5):
    """
    Recommends items for a user based on user-user collaborative filtering.
    """
    # Get similar users sorted by similarity score
    similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude the user themselves
    # Get items already interacted with by the target user
    target_user_items = set(interaction_matrix[user_id].nonzero()[1])
    item_scores = {}

    # Score items based on similarity scores of similar users
    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]
        similar_user_items = interaction_matrix[similar_user].nonzero()[1]

        for item in similar_user_items:
            if item not in target_user_items:  # Exclude items already interacted with
                item_scores[item] = item_scores.get(item, 0) + similarity_score

    # Sort items by their scores and return the top-k
    recommended_items = sorted(item_scores.keys(), key=lambda x: -item_scores[x])[:top_k]
    return recommended_items

# Example usage
user_id = 2  # Specify the user ID for whom recommendations are needed
recommended_items = recommend_items_user_user(user_id, interaction_matrix, user_similarity, top_k=5)
print("Collaborative Filtering Recommendations:", recommended_items)
