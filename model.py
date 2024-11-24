# import torch
# import numpy as np
# from transformers import CLIPProcessor, CLIPModel
# from datasets import load_dataset
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image
# import io
# from transformers import pipeline

# # Load combined embeddings
# all_embeddings = np.load('combined_embeddings.npy')

# # Normalize all_embeddings for cosine similarity
# all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

# # Load the dataset (e.g., WikiArt for training data)
# ds = load_dataset("Artificio/WikiArt")
# train_data = ds['train']

# # Load CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Check if GPU is available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # Load the input image using PIL
# image_path = "searchtest.jpg"  # Replace with your actual image path
# image = Image.open(image_path)

# # Example user text input
# text = "Give me the images with same style as the given image and with more aligned towards nature and contain green and blue colors"

# # Preprocess inputs
# inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
# inputs = {key: value.to(device) for key, value in inputs.items()}

# # Get the embeddings for the image and text
# with torch.no_grad():
#     image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
#     text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# # Normalize the embeddings
# image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# # Combine the embeddings with weights
# weight_img = 0.5
# weight_text = 0.5

# # Convert to CPU and NumPy
# image_features = image_features.cpu().detach().numpy()
# text_features = text_features.cpu().detach().numpy()

# # Compute the final embedding as a weighted combination of image and text embeddings
# final_embedding = weight_img * image_features + weight_text * text_features

# # Debug shapes
# print("Shape of final_embedding:", final_embedding.shape)
# print("Shape of all_embeddings:", all_embeddings.shape)

# # Adjust dimensions if necessary
# if final_embedding.shape[1] > all_embeddings.shape[1]:
#     # Trim final_embedding
#     final_embedding = final_embedding[:, :all_embeddings.shape[1]]
# elif final_embedding.shape[1] < all_embeddings.shape[1]:
#     # Pad final_embedding
#     padding_size = all_embeddings.shape[1] - final_embedding.shape[1]
#     padding = np.zeros((final_embedding.shape[0], padding_size))
#     final_embedding = np.hstack((final_embedding, padding))

# # Compute cosine similarity
# similarities = cosine_similarity(final_embedding, all_embeddings)
# print("Cosine similarities:", similarities)

# top_n_indices = np.argsort(similarities[0])[::-1][:10]

# # Recommend the top-N artworks
# recommended_artworks = [i for i in top_n_indices]

# print(recommended_artworks)

# def display_image(image_data):
#     if isinstance(image_data, Image.Image):  # PIL Image
#         image_data.show()
#     elif isinstance(image_data, bytes):  # Encoded image (e.g., byte string)
#         new_image = Image.open(io.BytesIO(image_data))
#         new_image.show()
#     elif isinstance(image_data, str):  # Path to the image file
#         new_image = Image.open(image_data)
#         new_image.show()
#     else:
#         print("Image format not recognized!")


# # Display images for the recommended indices
# for i in recommended_artworks:
#     # Get image data from the dataset
#     curr_img_data = ds['train'][int(i)]['image']
#     display_image(curr_img_data)

# # Display the user-provided input image
# display_image(image_path)  # Pass the input image path


# explanation_model = pipeline("text-generation", model="gpt-neo")

# def generate_explanation(input_text, curr_metadata, similarity_score):
#     """
#     Generate an explanation for the recommendation.
    
#     Args:
#         input_text (str): The user's preference text.
#         curr_metadata (dict): Metadata of the recommended artwork (e.g., artist, style, genre).
#         similarity_score (float): The cosine similarity score.

#     Returns:
#         str: Explanation text.
#     """
#     prompt = (
#         f"User's preference: {input_text}\n"
#         f"Recommended artwork details:\n"
#         f"- Artist: {curr_metadata['artist']}\n"
#         f"- Style: {curr_metadata['style']}\n"
#         f"- Genre: {curr_metadata['genre']}\n"
#         f"Similarity score: {similarity_score:.2f}\n"
#         "Explain why this artwork was recommended in a friendly and informative way:"
#     )
    
#     # Generate explanation using the LLM
#     response = explanation_model(prompt, max_length=100, num_return_sequences=1)
#     explanation = response[0]['generated_text']
#     return explanation

# for i in recommended_artworks:
#     # Get metadata for the current artwork
#     curr_metadata = {
#         "artist": ds['train'][int(i)]['artist'],
#         "style": ds['train'][int(i)]['style'],
#         "genre": ds['train'][int(i)]['genre']
#     }
#     similarity_score = similarities[0][int(i)]  # Get similarity score
    
#     # Generate and display explanation
#     explanation = generate_explanation(text, curr_metadata, similarity_score)
#     print(f"Recommendation {i}:")
#     print(f"Explanation: {explanation}")
    
#     # Display the recommended image
#     curr_img_data = ds['train'][int(i)]['image']
#     display_image(curr_img_data)



import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import requests
from tqdm import tqdm

def display_image(image_data):
    """
    Display an image from various possible formats.

    Args:
        image_data: Can be a PIL Image, bytes, a file path, or a URL.
    """
    try:
        if isinstance(image_data, Image.Image):  # PIL Image
            image_data.show()
        elif isinstance(image_data, bytes):  # Encoded image (e.g., byte string)
            new_image = Image.open(io.BytesIO(image_data))
            new_image.show()
        elif isinstance(image_data, str):
            if image_data.startswith('http://') or image_data.startswith('https://'):
                # If the string is a URL, download the image
                response = requests.get(image_data)
                new_image = Image.open(io.BytesIO(response.content))
                new_image.show()
            else:
                # Assume it's a local file path
                new_image = Image.open(image_data)
                new_image.show()
        else:
            print("Unsupported image format!")
    except Exception as e:
        print(f"Error displaying image: {e}")

def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
    """
    Generate a caption for the given image using the BLIP model.

    Args:
        image (PIL.Image.Image): The input image.
        blip_model: The BLIP model for image captioning.
        blip_processor: The BLIP processor for preprocessing.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        str: Generated caption for the image.
    """
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs,max_new_tokens=max_new_tokens)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
    """
    Generate an explanation for the recommendation based on similarity scores.

    Args:
        user_text (str): The user's preference text.
        curr_metadata (dict): Metadata of the recommended artwork.
        sim_image (float): Cosine similarity with the input image.
        sim_text (float): Cosine similarity with the input text.

    Returns:
        str: Explanation text.
    """
    # Determine the primary contributor to similarity
    margin = 0.05  # Threshold to determine significance
    if sim_image > sim_text + margin:
        reason = "the style and composition of the input image."
    elif sim_text > sim_image + margin:
        reason = "your textual preferences for nature and the specified colors."
    else:
        reason = "a balanced combination of both your image and textual preferences."

    explanation = (
        f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
        f"is recommended {reason} "
        f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
    )
    return explanation

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load combined embeddings
    try:
        all_embeddings = np.load('combined_embeddings.npy')
        print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
    except FileNotFoundError:
        print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
        return

    # Normalize all_embeddings for cosine similarity
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    print("Normalized all_embeddings for cosine similarity.")

    # Load the dataset (e.g., WikiArt for training data)
    try:
        ds = load_dataset("Artificio/WikiArt")
        train_data = ds['train']
        print("Loaded WikiArt dataset successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load CLIP model and processor
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.to(device)
        clip_model.eval()
        print("Loaded CLIP model and processor successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return

    # Load BLIP model and processor for image captioning
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.to(device)
        blip_model.eval()
        print("Loaded BLIP model and processor successfully.")
    except Exception as e:
        print(f"Error loading BLIP model: {e}")
        return

    # Load the input image using PIL
    image_path = "searchtest.jpg"  # Replace with your actual image path
    try:
        input_image = Image.open(image_path).convert("RGB")
        print(f"Loaded input image from {image_path}.")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Generate caption for the input image to provide context
    image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
    print(f"Generated image caption: {image_caption}")

    # Example user text input
    user_text = "Find similar images but with these flowers placed in a forest and in nature green land with dark theme."

    # Combine image caption with user text to make it context-aware
    context_aware_text = f"The given image is {image_caption}. {user_text}"
    print(f"Context-aware text: {context_aware_text}")

    # Preprocess inputs for CLIP
    inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("Preprocessed inputs for CLIP.")

    # Get the embeddings for the image and text
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    print("Generated image and text features using CLIP.")

    # Normalize the embeddings
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    print("Normalized image and text features.")

    # Combine the embeddings with weights
    weight_img = 0.1
    weight_text = 0.9

    # Convert to CPU and NumPy
    image_features_np = image_features.cpu().detach().numpy()
    text_features_np = text_features.cpu().detach().numpy()

    # Compute the final embedding as a weighted combination of image and text embeddings
    final_embedding = weight_img * image_features_np + weight_text * text_features_np

    # Normalize the final embedding
    final_embedding = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
    print("Computed final combined embedding.")

    # Debug shapes
    print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, 512)
    print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (103250, 515)

    # Ensure that the embedding dimensions match by padding
    embedding_dim = all_embeddings.shape[1]  # 515
    if final_embedding.shape[1] != embedding_dim:
        print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
        if final_embedding.shape[1] > embedding_dim:
            # Trim final_embedding
            final_embedding = final_embedding[:, :embedding_dim]
            print("Trimmed final_embedding.")
        else:
            # Calculate the number of zeros to pad
            padding_size = embedding_dim - final_embedding.shape[1]  # 3
            # Create padding of zeros
            padding = np.zeros((final_embedding.shape[0], padding_size))
            # Horizontally stack the padding to final_embedding
            final_embedding = np.hstack((final_embedding, padding))
            print(f"Padded final_embedding with {padding_size} zeros.")
        print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, 515)

    # **Pad image_features_np and text_features_np to 515 dimensions**
    if image_features_np.shape[1] != embedding_dim:
        print(f"Padded image_features_np from {image_features_np.shape[1]} to {embedding_dim} dimensions.")
        image_padding_size = embedding_dim - image_features_np.shape[1]  # 3
        padding_image = np.zeros((image_features_np.shape[0], image_padding_size))
        image_features_padded = np.hstack((image_features_np, padding_image))
    else:
        image_features_padded = image_features_np

    if text_features_np.shape[1] != embedding_dim:
        print(f"Padded text_features_np from {text_features_np.shape[1]} to {embedding_dim} dimensions.")
        text_padding_size = embedding_dim - text_features_np.shape[1]  # 3
        padding_text = np.zeros((text_features_np.shape[0], text_padding_size))
        text_features_padded = np.hstack((text_features_np, padding_text))
    else:
        text_features_padded = text_features_np

    # Compute cosine similarity between final_embedding and all_embeddings
    similarities = cosine_similarity(final_embedding, all_embeddings)
    print("Computed cosine similarities between the final embedding and all dataset embeddings.")

    # Get top-N indices based on similarity scores
    top_n = 6
    top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
    print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

    # Recommend the top-N artworks
    recommended_artworks = [int(i) for i in top_n_indices]

    # Display the user-provided input image
    print("\nDisplaying your input image:")
    display_image(image_path)

    # Iterate over the recommended artworks and display them with explanations
    for rank, i in enumerate(recommended_artworks, start=1):
        # Get the recommended artwork data
        try:
            artwork = train_data[i]
        except IndexError:
            print(f"Index {i} is out of bounds for the dataset.")
            continue

        # Extract metadata with default values if missing
        curr_metadata = {
            "artist": artwork.get('artist', 'Unknown Artist'),
            "style": artwork.get('style', 'Unknown Style'),
            "genre": artwork.get('genre', 'Unknown Genre')
        }

        # Get the embedding for the current artwork
        artwork_embedding = all_embeddings[i].reshape(1, -1)

        # **Compute similarity with padded image_features and text_features**
        sim_image = cosine_similarity(image_features_padded, artwork_embedding)[0][0]
        sim_text = cosine_similarity(text_features_padded, artwork_embedding)[0][0]

        # Generate explanation
        explanation = generate_explanation(user_text, curr_metadata, sim_image, sim_text)

        # Display the recommended image and its explanation
        print(f"\nRecommendation {rank}:")
        print(f"Index: {i}")
        print(f"Artist: {curr_metadata['artist']}")
        print(f"Style: {curr_metadata['style']}")
        print(f"Genre: {curr_metadata['genre']}")
        print(f"Explanation: {explanation}")

        # Display the image
        artwork_image = artwork['image']
        display_image(artwork_image)

if __name__ == "__main__":
    main()

