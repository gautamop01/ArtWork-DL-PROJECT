import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define helper functions
def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
    """
    Combines the current embedding with the previous one using a weighted average.
    """
    if previous_embedding is None:
        return current_embedding
    return weight * current_embedding + (1 - weight) * previous_embedding

def recommend_similar_artworks(combined_embedding, all_embeddings, k=5):
    """
    Recommends the top-k similar artworks based on cosine similarity.
    """
    similarities = cosine_similarity([combined_embedding], all_embeddings)
    top_k_indices = similarities.argsort()[0][-k:][::-1]  # Get indices of top-k most similar
    return top_k_indices

# Example data setup
# Create example embeddings for demonstration purposes
np.random.seed(42)  # Set seed for reproducibility
num_artworks = 100  # Number of artworks in the dataset
embedding_dim = 128  # Dimensionality of each embedding

# Randomly generate a matrix of embeddings for artworks
all_embeddings = np.random.rand(num_artworks, embedding_dim)

# Simulate a current embedding (e.g., for a selected artwork)
current_embedding = all_embeddings[10]  # Assume artwork at index 10 is the current one

# Initialize previous combined embedding as None
previous_combined_embedding = None
weight = 0.7  # Define the weight for combining embeddings

# Combine embeddings
combined_embedding = combine_embeddings_for_recommendation(
    current_embedding, previous_combined_embedding, weight
)
# Update the previous_combined_embedding
previous_combined_embedding = combined_embedding

# Get recommendations
recommended_indices = recommend_similar_artworks(combined_embedding, all_embeddings)
print("Content-Based Filtering Recommendations:", recommended_indices)
