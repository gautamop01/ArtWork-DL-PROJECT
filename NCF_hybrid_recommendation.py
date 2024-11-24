import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import io
from scipy.sparse import csr_matrix
import faiss
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from gensim.models import Word2Vec

ds = load_dataset("Artificio/WikiArt")

train_data = ds['train']

all_embeddings = np.load('combined_embeddings.npy')

all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

"""## Content-Based Filtering"""

def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
    """
    Combine embeddings using a weighted approach.

    Args:
        current_embedding (np.array): Embedding of the current artwork.
        previous_embedding (np.array): Combined embedding from previous interactions, or None.
        weight (float): Weight given to the current embedding (0 <= weight <= 1).

    Returns:
        np.array: Weighted combined embedding.
    """
    if previous_embedding is None:
        # No previous embedding exists
        return current_embedding
    else:
        # Combine embeddings using weights
        return weight * current_embedding + (1 - weight) * previous_embedding

def recommend_similar_artworks(combined_embedding, all_embeddings, k=5):
    """
    Recommend artworks similar to the combined embedding.

    Args:
        combined_embedding (np.array): Weighted combined embedding for recommendation.
        all_embeddings (np.array): All artwork embeddings in the dataset.
        k (int): Number of recommendations to return.

    Returns:
        list: Indices of the top-k recommended artworks.
    """
    # Compute cosine similarity
    similarities = cosine_similarity([combined_embedding], all_embeddings)
    top_k_indices = similarities.argsort()[0][-k:][::-1]  # Top-k most similar
    return top_k_indices

# Example usage
# Assume `current_embedding` is the embedding of the clicked artwork
# `previous_combined_embedding` is the embedding combined from prior interactions (initially None)
# `all_embeddings` contains all the artwork embeddings in the dataset

# User clicks on an artwork
current_embedding = all_embeddings[10]  # Example: clicked artwork's embedding
previous_combined_embedding = None  # Initially, no previous embedding exists

# Combine embeddings
weight = 0.7  # Example: give 70% weight to the current click
combined_embedding = combine_embeddings_for_recommendation(
    current_embedding, previous_combined_embedding, weight
)

# Update the previous combined embedding for future interactions
previous_combined_embedding = combined_embedding

# # Recommend similar artworks
# recommended_indices = recommend_similar_artworks(combined_embedding, all_embeddings)
# # print("Recommended Artwork Indices:", recommended_indices)

# Access the first data point from the training set
current_data_point = ds['train'][10]

# Check the column name that contains the image
image_column_name = "image"  # Replace with the actual column name if different

# Get the image from the dataset
image_data = current_data_point[image_column_name]

"""## Collaborative Filtering"""

user_interactions = [
    [0, 1],  # User 0 interacted with data point 1
    [0, 2],  # User 0 interacted with data point 2
    [1, 1],  # User 1 interacted with data point 0
    [1, 2],  # User 1 interacted with data point 3
    [2, 1],  # User 2 interacted with data point 4
    [3, 100],
    [4, 100],
    [3, 101],
    [4, 101],
    [3, 102]
]

user_ids = [interaction[0] for interaction in user_interactions]
data_point_indices = [interaction[1] for interaction in user_interactions]

n_users = max(user_ids) + 1  # Total number of users
n_items = all_embeddings.shape[0]  # Total number of data points from embeddings

data = [1] * len(user_interactions)  # Interaction weights (all set to 1 here)
interaction_matrix = csr_matrix((data, (user_ids, data_point_indices)), shape=(n_users, n_items))

# Compute user-user similarity using cosine similarity
user_similarity = cosine_similarity(interaction_matrix)

def recommend_items_user_user(user_id, interaction_matrix, user_similarity, top_k=5):
    """
    Recommend items to a user based on user-user collaborative filtering.

    Args:
        user_id: ID of the user to recommend items for.
        interaction_matrix: Sparse matrix of user-item interactions.
        user_similarity: User similarity matrix.
        top_k: Number of recommendations to generate.

    Returns:
        List of recommended item indices.
    """
    # Get user similarity scores for the given user
    similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude self (at index 0)

    # Get indices of items interacted by the target user
    target_user_items = set(interaction_matrix[user_id].nonzero()[1])

    # Keep track of item scores
    item_scores = {}

    # Aggregate scores from similar users
    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]

        # Get items interacted by the similar user
        similar_user_items = interaction_matrix[similar_user].nonzero()[1]

        for item in similar_user_items:
            if item not in target_user_items:  # Exclude already interacted items
                if item not in item_scores:
                    item_scores[item] = 0
                item_scores[item] += similarity_score

    # Sort items by aggregated score and return top_k recommendations
    recommended_items = sorted(item_scores.keys(), key=lambda x: -item_scores[x])[:top_k]
    return recommended_items

"""## Collaborative Filtering using NCF"""

class ArtworkDataset():
    def __init__(self, user_item_matrix, item_embeddings):
        """
        user_item_matrix: List of tuples [(user_id, item_id), ...]
        item_embeddings: Tensor of shape (num_items, combined_embedding_dim) -> Precomputed combined embeddings
        """
        self.user_item_matrix = user_item_matrix
        self.item_embeddings = item_embeddings

    def __len__(self):
        return len(self.user_item_matrix)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_matrix[idx]
        item_embedding = self.item_embeddings[item_id]
        return user_id, item_embedding

class NCFModel(nn.Module):
    def __init__(self, num_users, embedding_dim, combined_embedding_dim, hidden_layers):
        super(NCFModel, self).__init__()
        # Embedding layer for users (initialize with max number of users)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # MLP for interaction modeling
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + combined_embedding_dim, hidden_layers[0]),
            nn.ReLU(),
            *[layer for hidden_dim in zip(hidden_layers[:-1], hidden_layers[1:])
              for layer in (nn.Linear(hidden_dim[0], hidden_dim[1]), nn.ReLU())]
        )

        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_embedding):
        user_vec = self.user_embedding(user_id)
        combined = torch.cat([user_vec, item_embedding], dim=-1)
        hidden = self.mlp(combined)
        return self.sigmoid(self.output(hidden))

    def resize_user_embeddings(self, new_num_users):
        """
        Resize the user embedding layer to accommodate new users.
        """
        old_weights = self.user_embedding.weight.data
        self.user_embedding = nn.Embedding(new_num_users, old_weights.size(1))
        with torch.no_grad():
            self.user_embedding.weight[:old_weights.size(0)] = old_weights

def get_num_users(user_item_matrix):
    return max(user_id for user_id, _ in user_item_matrix) + 1

embedding_dim = 64
hidden_layers = [128, 64, 32]
batch_size = 128
learning_rate = 0.001
epochs = 10

user_item_matrix = [
    [0, 1],  # User 0 interacted with data point 1
    [0, 2],  # User 0 interacted with data point 2
    [1, 1],  # User 1 interacted with data point 0
    [1, 2],  # User 1 interacted with data point 3
    [2, 1],  # User 2 interacted with data point 4
    [3, 100],
    [4, 100],
    [3, 101],
    [4, 101],
    [4, 10]
]

all_embeddings_ncf = torch.tensor(all_embeddings, dtype=torch.float32)

num_users = get_num_users(user_item_matrix)

dataset_ncf = ArtworkDataset(user_item_matrix, all_embeddings_ncf)
data_loader = DataLoader(dataset_ncf, batch_size=batch_size, shuffle=True)

model = NCFModel(num_users, embedding_dim, all_embeddings_ncf.size(1), hidden_layers)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for user_id, item_embedding in data_loader:
        user_id = user_id.long()  # User IDs
        preds = model(user_id, item_embedding.float())
        labels = torch.ones_like(preds)  # Replace with actual labels if available
        loss = criterion(preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

new_user_id = 3
if new_user_id >= model.user_embedding.num_embeddings:
    # print("Resizing user embedding for new users.")
    model.resize_user_embeddings(new_user_id + 1)

# Inference Example
user_id = torch.tensor([0])  # Example user
item_embedding = all_embeddings_ncf[1]  # Example item embedding
prediction = model(user_id, item_embedding.unsqueeze(0))
# print(f"Recommendation score: {prediction.item()}")

def get_top_k_recommendations(user_id, model, all_embeddings, k=5):
    """
    user_id: The user for whom we want to get the recommendations
    model: The trained NCF model
    all_embeddings: The precomputed item embeddings (tensor)
    k: The number of recommendations to return (default 5)

    Returns:
    top_k_items_with_scores: List of tuples [(item_id, score), ...] for the top k recommended items
    """
    # Ensure user_id is a tensor
    user_id = torch.tensor([user_id], dtype=torch.long)

    # Get the user embedding from the model
    user_embedding = model.user_embedding(user_id)

    # Compute predicted scores for all items
    with torch.no_grad():
        scores = []
        for item_id in range(all_embeddings.shape[0]):  # Loop over all items
            item_embedding = all_embeddings[item_id].unsqueeze(0)  # Get embedding for the item
            score = model(user_id, item_embedding)  # Predict score for this item
            scores.append(score.item())  # Store the score

    # Convert scores to tensor
    scores = torch.tensor(scores)

    # Get the top k item indices and scores
    top_k_values, top_k_indices = torch.topk(scores, k)

    # Combine indices and their scores into a list of tuples
    top_k_items_with_scores = [(top_k_indices[i].item(), top_k_values[i].item()) for i in range(k)]

    return top_k_items_with_scores

"""## Hybrid Model using Content-Based filtering and 1st Collaborative Filtering approach"""

def hybrid_recommendation(user_id, combined_embedding, all_embeddings, interaction_matrix, user_similarity, content_weight=0.6, collaborative_weight=0.4, top_k=5):
    """
    Hybrid recommendation system combining content-based and collaborative filtering.

    Args:
        user_id: ID of the user to recommend items for.
        combined_embedding: Weighted combined embedding for content-based recommendation.
        all_embeddings: All artwork embeddings in the dataset.
        interaction_matrix: Sparse matrix of user-item interactions.
        user_similarity: User similarity matrix.
        content_weight: Weight for content-based recommendations.
        collaborative_weight: Weight for collaborative recommendations.
        top_k: Number of recommendations to generate.

    Returns:
        List of recommended item indices.
    """
    # Content-based recommendations
    content_similarities = cosine_similarity([combined_embedding], all_embeddings)
    content_scores = content_similarities[0]

    # Collaborative filtering recommendations
    similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude self (at index 0)
    target_user_items = set(interaction_matrix[user_id].nonzero()[1])
    collaborative_scores = np.zeros(all_embeddings.shape[0])

    # Aggregate scores from similar users
    for similar_user in similar_users:
        similarity_score = user_similarity[user_id, similar_user]
        similar_user_items = interaction_matrix[similar_user].nonzero()[1]

        for item in similar_user_items:
            if item not in target_user_items:  # Exclude already interacted items
                collaborative_scores[item] += similarity_score

    # Normalize both scores
    content_scores = content_scores / np.max(content_scores) if np.max(content_scores) > 0 else content_scores
    collaborative_scores = collaborative_scores / np.max(collaborative_scores) if np.max(collaborative_scores) > 0 else collaborative_scores

    # Combine scores using the specified weights
    final_scores = content_weight * content_scores + collaborative_weight * collaborative_scores

    # Get top-k recommendations
    recommended_items = np.argsort(-final_scores)[:top_k]
    return recommended_items

user_id = 2  # Example user ID
combined_embedding = previous_combined_embedding  # Use previously combined embedding
top_k = 5  # Number of recommendations

"""## Hybrid Model using Content-Based filtering and 2nd Collaborative Filtering approach"""

def hybrid_recommendations_ncf(user_id, model, all_embeddings, combined_embedding, k=5, content_weight=0.6, cf_weight=0.4):
    """
    Generate hybrid recommendations by combining content-based and collaborative filtering scores.

    Args:
        user_id: User ID for the collaborative filtering part.
        model: The trained NCF model for collaborative filtering.
        all_embeddings: Precomputed item embeddings (tensor).
        combined_embedding: Combined embedding from content-based filtering.
        k: Number of recommendations to return.
        content_weight: Weight for content-based filtering scores.
        cf_weight: Weight for collaborative filtering scores.

    Returns:
        List of tuples [(item_id, score)] for the top-k recommended items.
    """
    # Collaborative Filtering Scores
    user_id_tensor = torch.tensor([user_id], dtype=torch.long)
    with torch.no_grad():
        cf_scores = []
        for item_id in range(all_embeddings.shape[0]):
            item_embedding = all_embeddings[item_id].unsqueeze(0)
            score = model(user_id_tensor, item_embedding)
            cf_scores.append(score.item())
    cf_scores = np.array(cf_scores)

    # Content-Based Filtering Scores
    content_similarities = cosine_similarity([combined_embedding], all_embeddings.numpy())[0]

    # Hybrid Scores
    hybrid_scores = content_weight * content_similarities + cf_weight * cf_scores

    # Top-k Recommendations
    top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
    top_k_items_with_scores = [(idx, hybrid_scores[idx]) for idx in top_k_indices]

    return top_k_items_with_scores

user_id = 3  # Example user ID
current_embedding = all_embeddings_ncf[10]  # Example: embedding of the clicked artwork
previous_combined_embedding = None  # Start with no previous interaction
combined_embedding = combine_embeddings_for_recommendation(current_embedding, previous_combined_embedding, weight=0.7)

top_k_recommendations = hybrid_recommendations_ncf(
    user_id, model, all_embeddings_ncf, combined_embedding, k=5
)

print(f"Top 5 hybrid recommendations for user {user_id}:")
for item_id, score in top_k_recommendations:
    print(f"Item ID: {item_id}, Hybrid Score: {score:.4f}")