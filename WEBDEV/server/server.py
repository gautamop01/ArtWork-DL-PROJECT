# # # # # # # # # from flask import Flask, request, jsonify
# # # # # # # # # import pandas as pd
# # # # # # # # # import os
# # # # # # # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # # # # # # import jwt
# # # # # # # # # import datetime
# # # # # # # # # from functools import wraps
# # # # # # # # # from flask_cors import CORS
# # # # # # # # # from dotenv import load_dotenv
# # # # # # # # # import io
# # # # # # # # # from PIL import Image
# # # # # # # # # import numpy as np
# # # # # # # # # import torch
# # # # # # # # # from transformers import CLIPProcessor, CLIPModel
# # # # # # # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # # # # # # from datasets import load_dataset
# # # # # # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # # # # # import requests
# # # # # # # # # import base64
# # # # # # # # # from tqdm import tqdm

# # # # # # # # # def display_image(image_data):
# # # # # # # # #     """
# # # # # # # # #     Display an image from various possible formats.

# # # # # # # # #     Args:
# # # # # # # # #         image_data: Can be a PIL Image, bytes, a file path, or a URL.
# # # # # # # # #     """
# # # # # # # # #     try:
# # # # # # # # #         if isinstance(image_data, Image.Image):  # PIL Image
# # # # # # # # #             image_data.show()
# # # # # # # # #         elif isinstance(image_data, bytes):  # Encoded image (e.g., byte string)
# # # # # # # # #             new_image = Image.open(io.BytesIO(image_data))
# # # # # # # # #             new_image.show()
# # # # # # # # #         elif isinstance(image_data, str):
# # # # # # # # #             if image_data.startswith('http://') or image_data.startswith('https://'):
# # # # # # # # #                 # If the string is a URL, download the image
# # # # # # # # #                 response = requests.get(image_data)
# # # # # # # # #                 new_image = Image.open(io.BytesIO(response.content))
# # # # # # # # #                 new_image.show()
# # # # # # # # #             else:
# # # # # # # # #                 # Assume it's a local file path
# # # # # # # # #                 new_image = Image.open(image_data)
# # # # # # # # #                 new_image.show()
# # # # # # # # #         else:
# # # # # # # # #             print("Unsupported image format!")
# # # # # # # # #     except Exception as e:
# # # # # # # # #         print(f"Error displaying image: {e}")

# # # # # # # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # # # # # # #     """
# # # # # # # # #     Generate a caption for the given image using the BLIP model.

# # # # # # # # #     Args:
# # # # # # # # #         image (PIL.Image.Image): The input image.
# # # # # # # # #         blip_model: The BLIP model for image captioning.
# # # # # # # # #         blip_processor: The BLIP processor for preprocessing.
# # # # # # # # #         device (str): Device to run the model on ('cuda' or 'cpu').

# # # # # # # # #     Returns:
# # # # # # # # #         str: Generated caption for the image.
# # # # # # # # #     """
# # # # # # # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # # # # # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # # # # # # #     return caption

# # # # # # # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # # # # # # #     """
# # # # # # # # #     Generate an explanation for the recommendation based on similarity scores.

# # # # # # # # #     Args:
# # # # # # # # #         user_text (str): The user's preference text.
# # # # # # # # #         curr_metadata (dict): Metadata of the recommended artwork.
# # # # # # # # #         sim_image (float): Cosine similarity with the input image.
# # # # # # # # #         sim_text (float): Cosine similarity with the input text.

# # # # # # # # #     Returns:
# # # # # # # # #         str: Explanation text.
# # # # # # # # #     """
# # # # # # # # #     # Determine the primary contributor to similarity
# # # # # # # # #     margin = 0.05  # Threshold to determine significance
# # # # # # # # #     if sim_image > sim_text + margin:
# # # # # # # # #         reason = "the style and composition of the input image."
# # # # # # # # #     elif sim_text > sim_image + margin:
# # # # # # # # #         reason = "your textual preferences for nature and the specified colors."
# # # # # # # # #     else:
# # # # # # # # #         reason = "a balanced combination of both your image and textual preferences."

# # # # # # # # #     explanation = (
# # # # # # # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # # # # # # #         f"is recommended {reason} "
# # # # # # # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # # # # # # #     )
# # # # # # # # #     return explanation

# # # # # # # # # def encode_image_to_base64(image):
# # # # # # # # #     """
# # # # # # # # #     Encode a PIL Image to a Base64 string.

# # # # # # # # #     Args:
# # # # # # # # #         image (PIL.Image.Image): The image to encode.

# # # # # # # # #     Returns:
# # # # # # # # #         str: Base64 encoded string of the image.
# # # # # # # # #     """
# # # # # # # # #     buffered = io.BytesIO()
# # # # # # # # #     image.save(buffered, format="JPEG")
# # # # # # # # #     img_bytes = buffered.getvalue()
# # # # # # # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # # # # # # #     return img_base64

# # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # # print(f"Using device: {device}")
# # # # # # # # # # Load combined embeddings
# # # # # # # # # try:
# # # # # # # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # # # # # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # # # # # # except FileNotFoundError:
# # # # # # # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # # # # # # #     all_embeddings = None

# # # # # # # # # if all_embeddings is not None:
# # # # # # # # #     # Normalize all_embeddings for cosine similarity
# # # # # # # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # # # # # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # # # # # # else:
# # # # # # # # #     print("Skipping normalization due to missing embeddings.")

# # # # # # # # # # Load the dataset (e.g., WikiArt for training data)
# # # # # # # # # try:
# # # # # # # # #     ds = load_dataset("Artificio/WikiArt")
# # # # # # # # #     train_data = ds['train']
# # # # # # # # #     print("Loaded WikiArt dataset successfully.")
# # # # # # # # # except Exception as e:
# # # # # # # # #     print(f"Error loading dataset: {e}")
# # # # # # # # #     train_data = None

# # # # # # # # # # Load CLIP model and processor
# # # # # # # # # try:
# # # # # # # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # # # #     clip_model.to(device)
# # # # # # # # #     # clip_model.eval()
# # # # # # # # #     print("Loaded CLIP model and processor successfully.")
# # # # # # # # # except Exception as e:
# # # # # # # # #     print(f"Error loading CLIP model: {e}")
# # # # # # # # #     clip_model = None
# # # # # # # # #     clip_processor = None

# # # # # # # # # # Load BLIP model and processor for image captioning
# # # # # # # # # try:
# # # # # # # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # # # #     blip_model.to(device)
# # # # # # # # #     # blip_model.eval()
# # # # # # # # #     print("Loaded BLIP model and processor successfully.")
# # # # # # # # # except Exception as e:
# # # # # # # # #     print(f"Error loading BLIP model: {e}")
# # # # # # # # #     blip_model = None
# # # # # # # # #     blip_processor = None

# # # # # # # # # # Load environment variables from .env file
# # # # # # # # # load_dotenv()

# # # # # # # # # app = Flask(__name__)
# # # # # # # # # CORS(app)  # Enable CORS for all routes

# # # # # # # # # # Retrieve the secret key from environment variables
# # # # # # # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # # # # # # Ensure that the secret key is set
# # # # # # # # # if not app.config['SECRET_KEY']:
# # # # # # # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # # # # # DATABASE = 'users.xlsx'

# # # # # # # # # # Initialize the Excel file if it doesn't exist
# # # # # # # # # if not os.path.exists(DATABASE):
# # # # # # # # #     df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # # # # # # # #     df.to_excel(DATABASE, index=False)

# # # # # # # # # def load_users():
# # # # # # # # #     try:
# # # # # # # # #         return pd.read_excel(DATABASE)
# # # # # # # # #     except FileNotFoundError:
# # # # # # # # #         # If the file doesn't exist, create it
# # # # # # # # #         df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # # # # # # # #         df.to_excel(DATABASE, index=False)
# # # # # # # # #         return df
# # # # # # # # #     except Exception as e:
# # # # # # # # #         raise e

# # # # # # # # # def save_users(df):
# # # # # # # # #     try:
# # # # # # # # #         df.to_excel(DATABASE, index=False)
# # # # # # # # #     except Exception as e:
# # # # # # # # #         raise e

# # # # # # # # # def token_required(f):
# # # # # # # # #     @wraps(f)
# # # # # # # # #     def decorated(*args, **kwargs):
# # # # # # # # #         token = None

# # # # # # # # #         # JWT is passed in the request header
# # # # # # # # #         if 'Authorization' in request.headers:
# # # # # # # # #             auth_header = request.headers['Authorization']
# # # # # # # # #             # Header should be in the format 'Bearer <JWT>'
# # # # # # # # #             try:
# # # # # # # # #                 token = auth_header.split(" ")[1]
# # # # # # # # #             except IndexError:
# # # # # # # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # # # # # # #         if not token:
# # # # # # # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # # # # # # #         try:
# # # # # # # # #             # Decode the token to get the payload
# # # # # # # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # # # # # # #             current_user_email = data['email']
# # # # # # # # #         except jwt.ExpiredSignatureError:
# # # # # # # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # # # # # # #         except jwt.InvalidTokenError:
# # # # # # # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # # # # # # #         # Optionally, you can fetch the user from the database here
# # # # # # # # #         try:
# # # # # # # # #             users = load_users()
# # # # # # # # #         except Exception as e:
# # # # # # # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # # #         user = users[users['Email'] == current_user_email]
# # # # # # # # #         if user.empty:
# # # # # # # # #             return jsonify({'message': 'User not found!'}), 401

# # # # # # # # #         return f(current_user_email, *args, **kwargs)

# # # # # # # # #     return decorated

# # # # # # # # # @app.route('/signup', methods=['POST'])
# # # # # # # # # def signup():
# # # # # # # # #     data = request.get_json()
# # # # # # # # #     full_name = data.get('full_name')
# # # # # # # # #     email = data.get('email')
# # # # # # # # #     password = data.get('password')

# # # # # # # # #     if not all([full_name, email, password]):
# # # # # # # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # # # # # # #     try:
# # # # # # # # #         users = load_users()
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # # #     if email in users['Email'].values:
# # # # # # # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # # # # # # #     hashed_password = generate_password_hash(password)

# # # # # # # # #     new_user = pd.DataFrame({
# # # # # # # # #         'FullName': [full_name],
# # # # # # # # #         'Email': [email],
# # # # # # # # #         'Password': [hashed_password]
# # # # # # # # #     })

# # # # # # # # #     try:
# # # # # # # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # # # # # # #     try:
# # # # # # # # #         save_users(users)
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # # # # # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # # # # # # @app.route('/login', methods=['POST'])
# # # # # # # # # def login():
# # # # # # # # #     data = request.get_json()
# # # # # # # # #     email = data.get('email')
# # # # # # # # #     password = data.get('password')

# # # # # # # # #     if not all([email, password]):
# # # # # # # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # # # # # # #     try:
# # # # # # # # #         users = load_users()
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # # #     user = users[users['Email'] == email]

# # # # # # # # #     if user.empty:
# # # # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # # # #     stored_password = user.iloc[0]['Password']
# # # # # # # # #     full_name = user.iloc[0]['FullName']

# # # # # # # # #     if not check_password_hash(stored_password, password):
# # # # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # # # #     try:
# # # # # # # # #         # Generate JWT token
# # # # # # # # #         token = jwt.encode({
# # # # # # # # #             'email': email,
# # # # # # # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expires in 1 hour
# # # # # # # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # # # # # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # # # # # # @app.route('/protected', methods=['GET'])
# # # # # # # # # @token_required
# # # # # # # # # def protected_route(current_user_email):
# # # # # # # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # # # # # # --- New /chat Endpoint ---

# # # # # # # # # @app.route('/chat', methods=['POST'])
# # # # # # # # # @token_required  # Optional: Protect the chat endpoint
# # # # # # # # # def chat(current_user_email):
# # # # # # # # #     """
# # # # # # # # #     Handle chat requests with text and optional image.
# # # # # # # # #     Processes the inputs and returns a response.
# # # # # # # # #     """
# # # # # # # # #     # Extract 'text' from form data
# # # # # # # # #     text = request.form.get('text', '').strip()

# # # # # # # # #     # Extract 'image' from files, if provided
# # # # # # # # #     image_file = request.files.get('image', None)

# # # # # # # # #     image_data = None
# # # # # # # # #     if image_file:
# # # # # # # # #         try:
# # # # # # # # #             # Read image data into memory
# # # # # # # # #             image_bytes = image_file.read()
# # # # # # # # #             image = Image.open(io.BytesIO(image_bytes))
# # # # # # # # #             # Optionally, you can perform image preprocessing here
# # # # # # # # #             # For example, convert to RGB, resize, etc.
# # # # # # # # #             image = image.convert('RGB')
# # # # # # # # #             # Convert image to PIL Image
# # # # # # # # #             image_data = image
# # # # # # # # #         except Exception as e:
# # # # # # # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # # # # # # #     try:
# # # # # # # # #         # Call the predict function with text and image_data
# # # # # # # # #         result = predict(text, image_data)
# # # # # # # # #         return jsonify(result), 200
# # # # # # # # #     except Exception as e:
# # # # # # # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # # # # # # def predict(text, image_data=None):
# # # # # # # # #     """
# # # # # # # # #     Process the input text and image, generate recommendations,
# # # # # # # # #     and return them with explanations and metadata.

# # # # # # # # #     Args:
# # # # # # # # #         text (str): The text prompt from the user.
# # # # # # # # #         image_data (PIL.Image.Image, optional): The image data as a PIL Image.

# # # # # # # # #     Returns:
# # # # # # # # #         dict: A dictionary containing the response and recommended artworks.
# # # # # # # # #     """
# # # # # # # # #     if not all([
# # # # # # # # #         all_embeddings is not None, 
# # # # # # # # #         train_data is not None, 
# # # # # # # # #         clip_model is not None, 
# # # # # # # # #         clip_processor is not None, 
# # # # # # # # #         blip_model is not None, 
# # # # # # # # #         blip_processor is not None
# # # # # # # # #     ]):
# # # # # # # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # # # # # # #     input_image = image_data
# # # # # # # # #     user_text = text

# # # # # # # # #     # Generate caption for the input image to provide context
# # # # # # # # #     if input_image:
# # # # # # # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # # # # # # #         print(f"Generated image caption: {image_caption}")
# # # # # # # # #     else:
# # # # # # # # #         image_caption = ""

# # # # # # # # #     # Combine image caption with user text to make it context-aware
# # # # # # # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # # # # # # #     print(f"Context-aware text: {context_aware_text}")

# # # # # # # # #     # Preprocess inputs for CLIP
# # # # # # # # #     if input_image:
# # # # # # # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # # # # # # #     else:
# # # # # # # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # # # # # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # # # # # # #     print("Preprocessed inputs for CLIP.")

# # # # # # # # #     # Get the embeddings for the image and text
# # # # # # # # #     with torch.no_grad():
# # # # # # # # #         if input_image:
# # # # # # # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # # # # # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # # # #             image_features_np = image_features.cpu().detach().numpy()
# # # # # # # # #         else:
# # # # # # # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # # # # # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # # # # # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # # # #         text_features_np = text_features.cpu().detach().numpy()
# # # # # # # # #     print("Generated and normalized image and text features using CLIP.")

# # # # # # # # #     # Combine the embeddings with weights
# # # # # # # # #     weight_img = 0.1
# # # # # # # # #     weight_text = 0.9

# # # # # # # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # # # # # # #     final_embedding = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # # # # # # #     print("Computed final combined embedding.")

# # # # # # # # #     # Debug shapes
# # # # # # # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # # # # # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # # # # # # #     # Ensure that the embedding dimensions match by padding
# # # # # # # # #     embedding_dim = all_embeddings.shape[1]  # e.g., 515
# # # # # # # # #     if final_embedding.shape[1] != embedding_dim:
# # # # # # # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # # # # # # #         if final_embedding.shape[1] > embedding_dim:
# # # # # # # # #             # Trim final_embedding
# # # # # # # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # # # # # # #             print("Trimmed final_embedding.")
# # # # # # # # #         else:
# # # # # # # # #             # Calculate the number of zeros to pad
# # # # # # # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # # # # # # #             # Create padding of zeros
# # # # # # # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # # # # # # #             # Horizontally stack the padding to final_embedding
# # # # # # # # #             final_embedding = np.hstack((final_embedding, padding))
# # # # # # # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # # # # # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # # # # # # #     # Compute cosine similarity between final_embedding and all_embeddings
# # # # # # # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # # # # # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # # # # # # #     # Get top-N indices based on similarity scores
# # # # # # # # #     top_n = 6
# # # # # # # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # # # # # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # # # # # # #     # Recommend the top-N artworks
# # # # # # # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # # # # # # #     recommendations = []

# # # # # # # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # # # # # # #         # Get the recommended artwork data
# # # # # # # # #         try:
# # # # # # # # #             artwork = train_data[i]
# # # # # # # # #         except IndexError:
# # # # # # # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # # # # # # #             continue

# # # # # # # # #         # Extract metadata with default values if missing
# # # # # # # # #         curr_metadata = {
# # # # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # # # #             "genre": artwork.get('genre', 'Unknown Genre')
# # # # # # # # #         }

# # # # # # # # #         # Get the image data or URL from the dataset
# # # # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # # # #         # Fetch and encode the image
# # # # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # # # #             # If it's a URL, fetch the image via requests
# # # # # # # # #             try:
# # # # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # # # #                 if response.status_code == 200:
# # # # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # # # #                 else:
# # # # # # # # #                     print(f"Failed to fetch image from {image_data_or_url}, status code: {response.status_code}")
# # # # # # # # #                     artwork_image = None
# # # # # # # # #             except Exception as e:
# # # # # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # # # # #                 artwork_image = None
# # # # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # # # #             # If it's already a PIL Image, use it directly
# # # # # # # # #             artwork_image = image_data_or_url
# # # # # # # # #         else:
# # # # # # # # #             # Unsupported type
# # # # # # # # #             artwork_image = None

# # # # # # # # #         # Encode the image to Base64
# # # # # # # # #         if artwork_image:
# # # # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # # # #         else:
# # # # # # # # #             img_base64 = None

# # # # # # # # #         # Append the recommendation to the list
# # # # # # # # #         recommendations.append({
# # # # # # # # #             'rank': rank,
# # # # # # # # #             'index': i,
# # # # # # # # #             'artist': curr_metadata['artist'],
# # # # # # # # #             'style': curr_metadata['style'],
# # # # # # # # #             'genre': curr_metadata['genre'],
# # # # # # # # #             'image': img_base64  # Base64 encoded image
# # # # # # # # #         })

# # # # # # # # #     # Response text
# # # # # # # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # # # # # # #     return {
# # # # # # # # #         'response': response_text,
# # # # # # # # #         'recommendations': recommendations  # List of recommended artworks with metadata and images
# # # # # # # # #     }




# # # # # # # # # if __name__ == '__main__':
# # # # # # # # #     app.run(debug=True)


# # # # # # # # # backend/app.py

# # # # # # # # from flask import Flask, request, jsonify
# # # # # # # # import pandas as pd
# # # # # # # # import os
# # # # # # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # # # # # import jwt
# # # # # # # # import datetime
# # # # # # # # from functools import wraps
# # # # # # # # from flask_cors import CORS
# # # # # # # # from dotenv import load_dotenv
# # # # # # # # import io
# # # # # # # # from PIL import Image
# # # # # # # # import numpy as np
# # # # # # # # import torch
# # # # # # # # from transformers import CLIPProcessor, CLIPModel
# # # # # # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # # # # # from datasets import load_dataset
# # # # # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # # # # import requests
# # # # # # # # import base64
# # # # # # # # import json

# # # # # # # # def display_image(image_data):
# # # # # # # #     # Function to display images (not used in backend)
# # # # # # # #     pass

# # # # # # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # # # # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # # # # # #     with torch.no_grad():
# # # # # # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # # # # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # # # # # #     return caption

# # # # # # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # # # # # #     margin = 0.05
# # # # # # # #     if sim_image > sim_text + margin:
# # # # # # # #         reason = "the style and composition of the input image."
# # # # # # # #     elif sim_text > sim_image + margin:
# # # # # # # #         reason = "your textual preferences for nature and the specified colors."
# # # # # # # #     else:
# # # # # # # #         reason = "a balanced combination of both your image and textual preferences."

# # # # # # # #     explanation = (
# # # # # # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # # # # # #         f"is recommended {reason} "
# # # # # # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # # # # # #     )
# # # # # # # #     return explanation

# # # # # # # # def encode_image_to_base64(image):
# # # # # # # #     buffered = io.BytesIO()
# # # # # # # #     image.save(buffered, format="JPEG")
# # # # # # # #     img_bytes = buffered.getvalue()
# # # # # # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # # # # # #     return img_base64

# # # # # # # # def decode_embedding(embedding_str):
# # # # # # # #     return np.array(json.loads(embedding_str))

# # # # # # # # def encode_embedding(embedding_array):
# # # # # # # #     return json.dumps(embedding_array.tolist())

# # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # print(f"Using device: {device}")

# # # # # # # # # Load combined embeddings
# # # # # # # # try:
# # # # # # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # # # # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # # # # # except FileNotFoundError:
# # # # # # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # # # # # #     all_embeddings = None

# # # # # # # # if all_embeddings is not None:
# # # # # # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # # # # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # # # # # else:
# # # # # # # #     print("Skipping normalization due to missing embeddings.")

# # # # # # # # # Load the dataset (e.g., WikiArt for training data)
# # # # # # # # try:
# # # # # # # #     ds = load_dataset("Artificio/WikiArt")
# # # # # # # #     train_data = ds['train']
# # # # # # # #     print("Loaded WikiArt dataset successfully.")
# # # # # # # # except Exception as e:
# # # # # # # #     print(f"Error loading dataset: {e}")
# # # # # # # #     train_data = None

# # # # # # # # # Load CLIP model and processor
# # # # # # # # try:
# # # # # # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # # #     clip_model.to(device)
# # # # # # # #     print("Loaded CLIP model and processor successfully.")
# # # # # # # # except Exception as e:
# # # # # # # #     print(f"Error loading CLIP model: {e}")
# # # # # # # #     clip_model = None
# # # # # # # #     clip_processor = None

# # # # # # # # # Load BLIP model and processor for image captioning
# # # # # # # # try:
# # # # # # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # # #     blip_model.to(device)
# # # # # # # #     print("Loaded BLIP model and processor successfully.")
# # # # # # # # except Exception as e:
# # # # # # # #     print(f"Error loading BLIP model: {e}")
# # # # # # # #     blip_model = None
# # # # # # # #     blip_processor = None

# # # # # # # # # Load environment variables from .env file
# # # # # # # # load_dotenv()

# # # # # # # # app = Flask(__name__)
# # # # # # # # CORS(app)

# # # # # # # # # Retrieve the secret key from environment variables
# # # # # # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # # # # # Ensure that the secret key is set
# # # # # # # # if not app.config['SECRET_KEY']:
# # # # # # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # # # # DATABASE_USERS = 'users.xlsx'
# # # # # # # # DATABASE_LIKES = 'user_likes.xlsx'
# # # # # # # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'

# # # # # # # # # Initialize Excel files if they don't exist
# # # # # # # # for db_file, columns in [
# # # # # # # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # # # # # # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # # # # # # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding'])
# # # # # # # # ]:
# # # # # # # #     if not os.path.exists(db_file):
# # # # # # # #         df = pd.DataFrame(columns=columns)
# # # # # # # #         df.to_excel(db_file, index=False)
# # # # # # # #         print(f"Created {db_file} with columns: {columns}")

# # # # # # # # def load_users():
# # # # # # # #     try:
# # # # # # # #         return pd.read_excel(DATABASE_USERS)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def save_users(df):
# # # # # # # #     try:
# # # # # # # #         df.to_excel(DATABASE_USERS, index=False)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def load_user_likes():
# # # # # # # #     try:
# # # # # # # #         return pd.read_excel(DATABASE_LIKES)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def save_user_likes(df):
# # # # # # # #     try:
# # # # # # # #         df.to_excel(DATABASE_LIKES, index=False)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def load_user_embeddings():
# # # # # # # #     try:
# # # # # # # #         return pd.read_excel(DATABASE_EMBEDDINGS)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def save_user_embeddings(df):
# # # # # # # #     try:
# # # # # # # #         df.to_excel(DATABASE_EMBEDDINGS, index=False)
# # # # # # # #     except Exception as e:
# # # # # # # #         raise e

# # # # # # # # def token_required(f):
# # # # # # # #     @wraps(f)
# # # # # # # #     def decorated(*args, **kwargs):
# # # # # # # #         token = None

# # # # # # # #         if 'Authorization' in request.headers:
# # # # # # # #             auth_header = request.headers['Authorization']
# # # # # # # #             try:
# # # # # # # #                 token = auth_header.split(" ")[1]
# # # # # # # #             except IndexError:
# # # # # # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # # # # # #         if not token:
# # # # # # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # # # # # #         try:
# # # # # # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # # # # # #             current_user_email = data['email']
# # # # # # # #         except jwt.ExpiredSignatureError:
# # # # # # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # # # # # #         except jwt.InvalidTokenError:
# # # # # # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # # # # # #         try:
# # # # # # # #             users = load_users()
# # # # # # # #         except Exception as e:
# # # # # # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # #         user = users[users['Email'] == current_user_email]
# # # # # # # #         if user.empty:
# # # # # # # #             return jsonify({'message': 'User not found!'}), 401

# # # # # # # #         return f(current_user_email, *args, **kwargs)

# # # # # # # #     return decorated

# # # # # # # # @app.route('/signup', methods=['POST'])
# # # # # # # # def signup():
# # # # # # # #     data = request.get_json()
# # # # # # # #     full_name = data.get('full_name')
# # # # # # # #     email = data.get('email')
# # # # # # # #     password = data.get('password')

# # # # # # # #     if not all([full_name, email, password]):
# # # # # # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # # # # # #     try:
# # # # # # # #         users = load_users()
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # #     if email in users['Email'].values:
# # # # # # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # # # # # #     hashed_password = generate_password_hash(password)

# # # # # # # #     new_user = pd.DataFrame({
# # # # # # # #         'FullName': [full_name],
# # # # # # # #         'Email': [email],
# # # # # # # #         'Password': [hashed_password]
# # # # # # # #     })

# # # # # # # #     try:
# # # # # # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # # # # # #     try:
# # # # # # # #         save_users(users)
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # # # # # #     # Initialize user embedding with zeros
# # # # # # # #     try:
# # # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # # #             new_embedding = pd.DataFrame({
# # # # # # # #                 'UserEmail': [email],
# # # # # # # #                 'Embedding': [zero_embedding_encoded]
# # # # # # # #             })
# # # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # # #             print(f"Initialized zero embedding for user {email}.")
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # # # # # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # # # # # @app.route('/login', methods=['POST'])
# # # # # # # # def login():
# # # # # # # #     data = request.get_json()
# # # # # # # #     email = data.get('email')
# # # # # # # #     password = data.get('password')

# # # # # # # #     if not all([email, password]):
# # # # # # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # # # # # #     try:
# # # # # # # #         users = load_users()
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # # #     user = users[users['Email'] == email]

# # # # # # # #     if user.empty:
# # # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # # #     stored_password = user.iloc[0]['Password']
# # # # # # # #     full_name = user.iloc[0]['FullName']

# # # # # # # #     if not check_password_hash(stored_password, password):
# # # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # # #     try:
# # # # # # # #         token = jwt.encode({
# # # # # # # #             'email': email,
# # # # # # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # # # # # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # # # # # #     # Ensure user has an embedding; initialize if not
# # # # # # # #     try:
# # # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # # #             new_embedding = pd.DataFrame({
# # # # # # # #                 'UserEmail': [email],
# # # # # # # #                 'Embedding': [zero_embedding_encoded]
# # # # # # # #             })
# # # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # # #             print(f"Initialized zero embedding for user {email} on login.")
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # # # # # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # # # # # @app.route('/protected', methods=['GET'])
# # # # # # # # @token_required
# # # # # # # # def protected_route(current_user_email):
# # # # # # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # # # # # --- New Endpoints ---

# # # # # # # # @app.route('/get-images', methods=['GET'])
# # # # # # # # @token_required
# # # # # # # # def get_images(current_user_email):
# # # # # # # #     """
# # # # # # # #     Fetch a batch of images for the user.
# # # # # # # #     If the user has not liked any images, return random indices.
# # # # # # # #     Otherwise, fetch based on recommendations.
# # # # # # # #     """
# # # # # # # #     try:
# # # # # # # #         user_likes = load_user_likes()
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # # # # #     if not user_liked_images:
# # # # # # # #         # User hasn't liked any images yet, return random 30-40 images
# # # # # # # #         if train_data is not None:
# # # # # # # #             num_images = len(train_data)
# # # # # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # # # # #         else:
# # # # # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # # # # #     else:
# # # # # # # #         # Fetch recommended images based on user embedding
# # # # # # # #         try:
# # # # # # # #             user_embeddings = load_user_embeddings()
# # # # # # # #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # # #             if user_embedding_row.empty:
# # # # # # # #                 # Initialize embedding with zeros if not found
# # # # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # # #                 zero_embedding = np.zeros(embedding_dim)
# # # # # # # #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # # #                 new_embedding = pd.DataFrame({
# # # # # # # #                     'UserEmail': [current_user_email],
# # # # # # # #                     'Embedding': [zero_embedding_encoded]
# # # # # # # #                 })
# # # # # # # #                 user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # # #                 save_user_embeddings(user_embeddings)
# # # # # # # #                 user_embedding = zero_embedding.reshape(1, -1)
# # # # # # # #                 print(f"Initialized zero embedding for user {current_user_email} in /get-images.")
# # # # # # # #             else:
# # # # # # # #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
            
# # # # # # # #             # Ensure user_embedding has the correct dimension
# # # # # # # #             if all_embeddings is not None:
# # # # # # # #                 embedding_dim = all_embeddings.shape[1]
# # # # # # # #                 if user_embedding.shape[1] != embedding_dim:
# # # # # # # #                     if user_embedding.shape[1] > embedding_dim:
# # # # # # # #                         user_embedding = user_embedding[:, :embedding_dim]
# # # # # # # #                         print("Trimmed user_embedding to match embedding_dim.")
# # # # # # # #                     else:
# # # # # # # #                         padding_size = embedding_dim - user_embedding.shape[1]
# # # # # # # #                         padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # # # #                         user_embedding = np.hstack((user_embedding, padding))
# # # # # # # #                         print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # # # #                     # Update the embedding in the dataframe
# # # # # # # #                     user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # # # #                     user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # # # #                     save_user_embeddings(user_embeddings)
# # # # # # # #             else:
# # # # # # # #                 return jsonify({'message': 'No embeddings available for recommendation.'}), 500

# # # # # # # #             # Compute similarities
# # # # # # # #             similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # # # #             top_indices = similarities.argsort()[0][::-1]
# # # # # # # #             # Exclude already liked images
# # # # # # # #             recommended_indices = [i for i in top_indices if i not in user_liked_images]
# # # # # # # #             # Select top 10
# # # # # # # #             indices = recommended_indices[:10]
# # # # # # # #         except Exception as e:
# # # # # # # #             return jsonify({'message': f'Error fetching recommendations: {str(e)}'}), 500

# # # # # # # #     recommendations = []

# # # # # # # #     for idx in indices:
# # # # # # # #         try:
# # # # # # # #             artwork = train_data[idx]
# # # # # # # #         except IndexError:
# # # # # # # #             continue

# # # # # # # #         curr_metadata = {
# # # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # # #         }

# # # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # # #             try:
# # # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # # #                 if response.status_code == 200:
# # # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # # #                 else:
# # # # # # # #                     artwork_image = None
# # # # # # # #             except:
# # # # # # # #                 artwork_image = None
# # # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # # #             artwork_image = image_data_or_url
# # # # # # # #         else:
# # # # # # # #             artwork_image = None

# # # # # # # #         if artwork_image:
# # # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # # #         else:
# # # # # # # #             img_base64 = None

# # # # # # # #         recommendations.append({
# # # # # # # #             'index': idx,
# # # # # # # #             'artist': curr_metadata['artist'],
# # # # # # # #             'style': curr_metadata['style'],
# # # # # # # #             'genre': curr_metadata['genre'],
# # # # # # # #             'description': curr_metadata['description'],
# # # # # # # #             'image': img_base64
# # # # # # # #         })

# # # # # # # #     return jsonify({'images': recommendations}), 200

# # # # # # # # @app.route('/like-image', methods=['POST'])
# # # # # # # # @token_required
# # # # # # # # def like_image(current_user_email):
# # # # # # # #     """
# # # # # # # #     Records a user's like for an image and updates embeddings.
# # # # # # # #     """
# # # # # # # #     data = request.get_json()
# # # # # # # #     image_index = data.get('image_index')

# # # # # # # #     if image_index is None:
# # # # # # # #         return jsonify({'message': 'Image index is required.'}), 400

# # # # # # # #     # Record the like
# # # # # # # #     try:
# # # # # # # #         user_likes = load_user_likes()
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # # # #     new_like = pd.DataFrame({
# # # # # # # #         'UserEmail': [current_user_email],
# # # # # # # #         'ImageIndex': [image_index],
# # # # # # # #         'LikedAt': [datetime.datetime.utcnow()]
# # # # # # # #     })

# # # # # # # #     try:
# # # # # # # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # # # # # # #         save_user_likes(user_likes)
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # # # # # # #     # Update user embedding after k likes
# # # # # # # #     k = 5  # Define k as needed
# # # # # # # #     user_total_likes = user_likes[user_likes['UserEmail'] == current_user_email].shape[0]

# # # # # # # #     if user_total_likes % k == 0:
# # # # # # # #         try:
# # # # # # # #             user_embeddings = load_user_embeddings()
# # # # # # # #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # # #             if user_embedding_row.empty:
# # # # # # # #                 # Initialize embedding with zeros if not found
# # # # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # # #                 zero_embedding = np.zeros(embedding_dim)
# # # # # # # #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # # #                 new_user_embedding = pd.DataFrame({
# # # # # # # #                     'UserEmail': [current_user_email],
# # # # # # # #                     'Embedding': [zero_embedding_encoded]
# # # # # # # #                 })
# # # # # # # #                 user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # # # # # #                 save_user_embeddings(user_embeddings)
# # # # # # # #                 user_embedding = zero_embedding
# # # # # # # #                 print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # # # # # # #             else:
# # # # # # # #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # # # # # # #             # Fetch user's liked image embeddings
# # # # # # # #             liked_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()
# # # # # # # #             if all_embeddings is not None and len(liked_indices) > 0:
# # # # # # # #                 liked_embeddings = all_embeddings[liked_indices]
# # # # # # # #                 # Compute the new embedding as the average
# # # # # # # #                 new_embedding = np.mean(liked_embeddings, axis=0)
# # # # # # # #                 new_embedding = new_embedding / np.linalg.norm(new_embedding)
# # # # # # # #             else:
# # # # # # # #                 # If no embeddings available, use zero embedding
# # # # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # # #                 new_embedding = np.zeros(embedding_dim)

# # # # # # # #             # Save the new embedding
# # # # # # # #             if user_embedding_row.empty:
# # # # # # # #                 # Add new user embedding
# # # # # # # #                 new_user_embedding = pd.DataFrame({
# # # # # # # #                     'UserEmail': [current_user_email],
# # # # # # # #                     'Embedding': [encode_embedding(new_embedding)]
# # # # # # # #                 })
# # # # # # # #                 user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # # # # # #             else:
# # # # # # # #                 # Update existing embedding
# # # # # # # #                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(new_embedding)

# # # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # # #             print(f"Updated embedding for user {current_user_email} after {k} likes.")
# # # # # # # #         except Exception as e:
# # # # # # # #             return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # # # # # # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # # # # # # @app.route('/recommend-images', methods=['GET'])
# # # # # # # # @token_required
# # # # # # # # def recommend_images(current_user_email):
# # # # # # # #     """
# # # # # # # #     Provides personalized recommendations based on user embeddings.
# # # # # # # #     """
# # # # # # # #     try:
# # # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # # #         if user_embedding_row.empty:
# # # # # # # #             # Initialize embedding with zeros if not found
# # # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # # #             new_embedding = pd.DataFrame({
# # # # # # # #                 'UserEmail': [current_user_email],
# # # # # # # #                 'Embedding': [zero_embedding_encoded]
# # # # # # # #             })
# # # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # # # # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # # # # # # #         else:
# # # # # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
        
# # # # # # # #         # Ensure user_embedding has the correct dimension
# # # # # # # #         if all_embeddings is not None:
# # # # # # # #             embedding_dim = all_embeddings.shape[1]
# # # # # # # #             if user_embedding.shape[1] != embedding_dim:
# # # # # # # #                 if user_embedding.shape[1] > embedding_dim:
# # # # # # # #                     user_embedding = user_embedding[:, :embedding_dim]
# # # # # # # #                     print("Trimmed user_embedding to match embedding_dim.")
# # # # # # # #                 else:
# # # # # # # #                     padding_size = embedding_dim - user_embedding.shape[1]
# # # # # # # #                     padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # # # #                     user_embedding = np.hstack((user_embedding, padding))
# # # # # # # #                     print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # # # #                 # Update the embedding in the dataframe
# # # # # # # #                 user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # # # #                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # # # #                 save_user_embeddings(user_embeddings)
# # # # # # # #         else:
# # # # # # # #             return jsonify({'message': 'No embeddings available for recommendation.'}), 500

# # # # # # # #         # Compute similarities
# # # # # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # # # #         top_n = 10
# # # # # # # #         top_n_indices = similarities.argsort()[0][::-1][:top_n]
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error computing similarities: {str(e)}'}), 500

# # # # # # # #     recommendations = []

# # # # # # # #     for idx in top_n_indices:
# # # # # # # #         try:
# # # # # # # #             artwork = train_data[idx]
# # # # # # # #         except IndexError:
# # # # # # # #             continue

# # # # # # # #         curr_metadata = {
# # # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # # #         }

# # # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # # #             try:
# # # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # # #                 if response.status_code == 200:
# # # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # # #                 else:
# # # # # # # #                     artwork_image = None
# # # # # # # #             except:
# # # # # # # #                 artwork_image = None
# # # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # # #             artwork_image = image_data_or_url
# # # # # # # #         else:
# # # # # # # #             artwork_image = None

# # # # # # # #         if artwork_image:
# # # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # # #         else:
# # # # # # # #             img_base64 = None

# # # # # # # #         recommendations.append({
# # # # # # # #             'index': idx,
# # # # # # # #             'artist': curr_metadata['artist'],
# # # # # # # #             'style': curr_metadata['style'],
# # # # # # # #             'genre': curr_metadata['genre'],
# # # # # # # #             'description': curr_metadata['description'],
# # # # # # # #             'image': img_base64
# # # # # # # #         })

# # # # # # # #     return jsonify({'recommendations': recommendations}), 200

# # # # # # # # @app.route('/chat', methods=['POST'])
# # # # # # # # @token_required
# # # # # # # # def chat(current_user_email):
# # # # # # # #     """
# # # # # # # #     Handle chat requests with text and optional image.
# # # # # # # #     Processes the inputs and returns a response.
# # # # # # # #     """
# # # # # # # #     text = request.form.get('text', '').strip()
# # # # # # # #     image_file = request.files.get('image', None)

# # # # # # # #     image_data = None
# # # # # # # #     if image_file:
# # # # # # # #         try:
# # # # # # # #             image_bytes = image_file.read()
# # # # # # # #             image = Image.open(io.BytesIO(image_bytes))
# # # # # # # #             image = image.convert('RGB')
# # # # # # # #             image_data = image
# # # # # # # #         except Exception as e:
# # # # # # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # # # # # #     try:
# # # # # # # #         result = predict(text, image_data)
# # # # # # # #         return jsonify(result), 200
# # # # # # # #     except Exception as e:
# # # # # # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # # # # # def predict(text, image_data=None):
# # # # # # # #     """
# # # # # # # #     Process the input text and image, generate recommendations,
# # # # # # # #     and return them with explanations and metadata.
# # # # # # # #     """
# # # # # # # #     if not all([
# # # # # # # #         all_embeddings is not None, 
# # # # # # # #         train_data is not None, 
# # # # # # # #         clip_model is not None, 
# # # # # # # #         clip_processor is not None, 
# # # # # # # #         blip_model is not None, 
# # # # # # # #         blip_processor is not None
# # # # # # # #     ]):
# # # # # # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # # # # # #     input_image = image_data
# # # # # # # #     user_text = text

# # # # # # # #     if input_image:
# # # # # # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # # # # # #         print(f"Generated image caption: {image_caption}")
# # # # # # # #     else:
# # # # # # # #         image_caption = ""

# # # # # # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # # # # # #     print(f"Context-aware text: {context_aware_text}")

# # # # # # # #     if input_image:
# # # # # # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # # # # # #     else:
# # # # # # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # # # # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # # # # # #     print("Preprocessed inputs for CLIP.")

# # # # # # # #     with torch.no_grad():
# # # # # # # #         if input_image:
# # # # # # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # # # # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # # #             image_features_np = image_features.cpu().detach().numpy()
# # # # # # # #         else:
# # # # # # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # # # # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # # # # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # # #         text_features_np = text_features.cpu().detach().numpy()
# # # # # # # #     print("Generated and normalized image and text features using CLIP.")

# # # # # # # #     weight_img = 0.1
# # # # # # # #     weight_text = 0.9

# # # # # # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # # # # # #     final_embedding = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # # # # # #     print("Computed final combined embedding.")

# # # # # # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # # # # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # # # # # #     embedding_dim = all_embeddings.shape[1]
# # # # # # # #     if final_embedding.shape[1] != embedding_dim:
# # # # # # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # # # # # #         if final_embedding.shape[1] > embedding_dim:
# # # # # # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # # # # # #             print("Trimmed final_embedding.")
# # # # # # # #         else:
# # # # # # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # # # # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # # # # # #             final_embedding = np.hstack((final_embedding, padding))
# # # # # # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # # # # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # # # # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # # # # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # # # # # #     top_n = 10
# # # # # # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # # # # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # # # # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # # # # # #     recommendations = []

# # # # # # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # # # # # #         try:
# # # # # # # #             artwork = train_data[i]
# # # # # # # #         except IndexError:
# # # # # # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # # # # # #             continue

# # # # # # # #         curr_metadata = {
# # # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # # #         }

# # # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # # #             try:
# # # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # # #                 if response.status_code == 200:
# # # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # # #                 else:
# # # # # # # #                     artwork_image = None
# # # # # # # #             except Exception as e:
# # # # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # # # #                 artwork_image = None
# # # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # # #             artwork_image = image_data_or_url
# # # # # # # #         else:
# # # # # # # #             artwork_image = None

# # # # # # # #         if artwork_image:
# # # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # # #         else:
# # # # # # # #             img_base64 = None

# # # # # # # #         recommendations.append({
# # # # # # # #             'rank': rank,
# # # # # # # #             'index': i,
# # # # # # # #             'artist': curr_metadata['artist'],
# # # # # # # #             'style': curr_metadata['style'],
# # # # # # # #             'genre': curr_metadata['genre'],
# # # # # # # #             # 'description': curr_metadata['description'],
# # # # # # # #             'image': img_base64
# # # # # # # #         })

# # # # # # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # # # # # #     return {
# # # # # # # #         'response': response_text,
# # # # # # # #         'recommendations': recommendations
# # # # # # # #     }

# # # # # # # # if __name__ == '__main__':
# # # # # # # #     app.run(debug=True)


# # # # # # # # backend/app.py

# # # # # # # from flask import Flask, request, jsonify
# # # # # # # import pandas as pd
# # # # # # # import os
# # # # # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # # # # import jwt
# # # # # # # import datetime
# # # # # # # from functools import wraps
# # # # # # # from flask_cors import CORS
# # # # # # # from dotenv import load_dotenv
# # # # # # # import io
# # # # # # # from PIL import Image
# # # # # # # import numpy as np
# # # # # # # import torch
# # # # # # # from transformers import CLIPProcessor, CLIPModel
# # # # # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # # # # from datasets import load_dataset
# # # # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # # # import requests
# # # # # # # import base64
# # # # # # # import json

# # # # # # # def display_image(image_data):
# # # # # # #     # Function to display images (not used in backend)
# # # # # # #     pass

# # # # # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # # # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # # # # #     with torch.no_grad():
# # # # # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # # # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # # # # #     return caption

# # # # # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # # # # #     margin = 0.05
# # # # # # #     if sim_image > sim_text + margin:
# # # # # # #         reason = "the style and composition of the input image."
# # # # # # #     elif sim_text > sim_image + margin:
# # # # # # #         reason = "your textual preferences for nature and the specified colors."
# # # # # # #     else:
# # # # # # #         reason = "a balanced combination of both your image and textual preferences."

# # # # # # #     explanation = (
# # # # # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # # # # #         f"is recommended {reason} "
# # # # # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # # # # #     )
# # # # # # #     return explanation

# # # # # # # def encode_image_to_base64(image):
# # # # # # #     buffered = io.BytesIO()
# # # # # # #     image.save(buffered, format="JPEG")
# # # # # # #     img_bytes = buffered.getvalue()
# # # # # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # # # # #     return img_base64

# # # # # # # def decode_embedding(embedding_str):
# # # # # # #     return np.array(json.loads(embedding_str))

# # # # # # # def encode_embedding(embedding_array):
# # # # # # #     return json.dumps(embedding_array.tolist())

# # # # # # # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# # # # # # #     """
# # # # # # #     Combines the current embedding with the previous one using a weighted average.
# # # # # # #     """
# # # # # # #     if previous_embedding is None:
# # # # # # #         return current_embedding
# # # # # # #     return weight * current_embedding + (1 - weight) * previous_embedding

# # # # # # # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# # # # # # #     """
# # # # # # #     Recommends the top-k similar artworks based on cosine similarity.
# # # # # # #     """
# # # # # # #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# # # # # # #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# # # # # # #     return top_k_indices

# # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # print(f"Using device: {device}")

# # # # # # # # Load combined embeddings
# # # # # # # try:
# # # # # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # # # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # # # # except FileNotFoundError:
# # # # # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # # # # #     all_embeddings = None

# # # # # # # if all_embeddings is not None:
# # # # # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # # # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # # # # else:
# # # # # # #     print("Skipping normalization due to missing embeddings.")

# # # # # # # # Load the dataset (e.g., WikiArt for training data)
# # # # # # # try:
# # # # # # #     ds = load_dataset("Artificio/WikiArt")
# # # # # # #     train_data = ds['train']
# # # # # # #     print("Loaded WikiArt dataset successfully.")
# # # # # # # except Exception as e:
# # # # # # #     print(f"Error loading dataset: {e}")
# # # # # # #     train_data = None

# # # # # # # # Load CLIP model and processor
# # # # # # # try:
# # # # # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # # # # #     clip_model.to(device)
# # # # # # #     print("Loaded CLIP model and processor successfully.")
# # # # # # # except Exception as e:
# # # # # # #     print(f"Error loading CLIP model: {e}")
# # # # # # #     clip_model = None
# # # # # # #     clip_processor = None

# # # # # # # # Load BLIP model and processor for image captioning
# # # # # # # try:
# # # # # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # # #     blip_model.to(device)
# # # # # # #     print("Loaded BLIP model and processor successfully.")
# # # # # # # except Exception as e:
# # # # # # #     print(f"Error loading BLIP model: {e}")
# # # # # # #     blip_model = None
# # # # # # #     blip_processor = None

# # # # # # # # Load environment variables from .env file
# # # # # # # load_dotenv()

# # # # # # # app = Flask(__name__)
# # # # # # # CORS(app)

# # # # # # # # Retrieve the secret key from environment variables
# # # # # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # # # # Ensure that the secret key is set
# # # # # # # if not app.config['SECRET_KEY']:
# # # # # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # # # # Define Excel database files
# # # # # # # DATABASE_USERS = 'users.xlsx'
# # # # # # # DATABASE_LIKES = 'user_likes.xlsx'
# # # # # # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'

# # # # # # # # Initialize Excel files if they don't exist
# # # # # # # for db_file, columns in [
# # # # # # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # # # # # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # # # # # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # # # # ]:
# # # # # # #     if not os.path.exists(db_file):
# # # # # # #         df = pd.DataFrame(columns=columns)
# # # # # # #         df.to_excel(db_file, index=False)
# # # # # # #         print(f"Created {db_file} with columns: {columns}")

# # # # # # # def load_users():
# # # # # # #     try:
# # # # # # #         return pd.read_excel(DATABASE_USERS)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def save_users(df):
# # # # # # #     try:
# # # # # # #         df.to_excel(DATABASE_USERS, index=False)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def load_user_likes():
# # # # # # #     try:
# # # # # # #         return pd.read_excel(DATABASE_LIKES)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def save_user_likes(df):
# # # # # # #     try:
# # # # # # #         df.to_excel(DATABASE_LIKES, index=False)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def load_user_embeddings():
# # # # # # #     try:
# # # # # # #         return pd.read_excel(DATABASE_EMBEDDINGS)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def save_user_embeddings(df):
# # # # # # #     try:
# # # # # # #         df.to_excel(DATABASE_EMBEDDINGS, index=False)
# # # # # # #     except Exception as e:
# # # # # # #         raise e

# # # # # # # def token_required(f):
# # # # # # #     @wraps(f)
# # # # # # #     def decorated(*args, **kwargs):
# # # # # # #         token = None

# # # # # # #         if 'Authorization' in request.headers:
# # # # # # #             auth_header = request.headers['Authorization']
# # # # # # #             try:
# # # # # # #                 token = auth_header.split(" ")[1]
# # # # # # #             except IndexError:
# # # # # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # # # # #         if not token:
# # # # # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # # # # #         try:
# # # # # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # # # # #             current_user_email = data['email']
# # # # # # #         except jwt.ExpiredSignatureError:
# # # # # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # # # # #         except jwt.InvalidTokenError:
# # # # # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # # # # #         try:
# # # # # # #             users = load_users()
# # # # # # #         except Exception as e:
# # # # # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # #         user = users[users['Email'] == current_user_email]
# # # # # # #         if user.empty:
# # # # # # #             return jsonify({'message': 'User not found!'}), 401

# # # # # # #         return f(current_user_email, *args, **kwargs)

# # # # # # #     return decorated

# # # # # # # @app.route('/signup', methods=['POST'])
# # # # # # # def signup():
# # # # # # #     data = request.get_json()
# # # # # # #     full_name = data.get('full_name')
# # # # # # #     email = data.get('email')
# # # # # # #     password = data.get('password')

# # # # # # #     if not all([full_name, email, password]):
# # # # # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # # # # #     try:
# # # # # # #         users = load_users()
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # #     if email in users['Email'].values:
# # # # # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # # # # #     hashed_password = generate_password_hash(password)

# # # # # # #     new_user = pd.DataFrame({
# # # # # # #         'FullName': [full_name],
# # # # # # #         'Email': [email],
# # # # # # #         'Password': [hashed_password]
# # # # # # #     })

# # # # # # #     try:
# # # # # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # # # # #     try:
# # # # # # #         save_users(users)
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # # # # #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# # # # # # #     try:
# # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # #             new_embedding = pd.DataFrame({
# # # # # # #                 'UserEmail': [email],
# # # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # # #                 'LastRecommendedIndex': [0],
# # # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # # #             })
# # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # #             print(f"Initialized zero embedding for user {email}.")
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # # # # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # # # # @app.route('/login', methods=['POST'])
# # # # # # # def login():
# # # # # # #     data = request.get_json()
# # # # # # #     email = data.get('email')
# # # # # # #     password = data.get('password')

# # # # # # #     if not all([email, password]):
# # # # # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # # # # #     try:
# # # # # # #         users = load_users()
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # # #     user = users[users['Email'] == email]

# # # # # # #     if user.empty:
# # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # #     stored_password = user.iloc[0]['Password']
# # # # # # #     full_name = user.iloc[0]['FullName']

# # # # # # #     if not check_password_hash(stored_password, password):
# # # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # # #     try:
# # # # # # #         token = jwt.encode({
# # # # # # #             'email': email,
# # # # # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # # # # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # # # # #     # Ensure user has an embedding; initialize if not
# # # # # # #     try:
# # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # #             new_embedding = pd.DataFrame({
# # # # # # #                 'UserEmail': [email],
# # # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # # #                 'LastRecommendedIndex': [0],
# # # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # # #             })
# # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # #             print(f"Initialized zero embedding for user {email} on login.")
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # # # # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # # # # @app.route('/protected', methods=['GET'])
# # # # # # # @token_required
# # # # # # # def protected_route(current_user_email):
# # # # # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # # # # --- New Endpoints ---

# # # # # # # @app.route('/get-images', methods=['GET'])
# # # # # # # @token_required
# # # # # # # def get_images(current_user_email):
# # # # # # #     """
# # # # # # #     Fetch a batch of images for the user.
# # # # # # #     If the user has not liked any images, return random indices.
# # # # # # #     Otherwise, fetch based on recommendations.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         user_likes = load_user_likes()
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # # # #     try:
# # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # #         if user_embedding_row.empty:
# # # # # # #             # Initialize embedding with zeros if not found
# # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # #             new_embedding = pd.DataFrame({
# # # # # # #                 'UserEmail': [current_user_email],
# # # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # # #                 'LastRecommendedIndex': [0],
# # # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # # #             })
# # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # # # #             print(f"Initialized zero embedding for user {current_user_email} in /get-images.")
# # # # # # #         else:
# # # # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # # # #     # Check if user has liked any images
# # # # # # #     if not user_liked_images:
# # # # # # #         # User hasn't liked any images yet, return random 40 images
# # # # # # #         if train_data is not None:
# # # # # # #             num_images = len(train_data)
# # # # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # # # #         else:
# # # # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # # # #     else:
# # # # # # #         if all_embeddings is None:
# # # # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # # # #         # Ensure user_embedding has the correct dimension
# # # # # # #         embedding_dim = all_embeddings.shape[1]
# # # # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # # # #             else:
# # # # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # # #             # Update the embedding in the dataframe
# # # # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # # # # # #             save_user_embeddings(user_embeddings)

# # # # # # #         # Compute similarities
# # # # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # # # #         # Exclude already liked images
# # # # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # # # #         # Fetch LastRecommendedIndex
# # # # # # #         try:
# # # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # # #         except:
# # # # # # #             last_recommended_index = 0

# # # # # # #         # Define batch size
# # # # # # #         batch_size = 10

# # # # # # #         # Select the next batch
# # # # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # # # #         # Update LastRecommendedIndex
# # # # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # # # #         save_user_embeddings(user_embeddings)

# # # # # # #     recommendations = []

# # # # # # #     for idx in indices:
# # # # # # #         try:
# # # # # # #             artwork = train_data[int(idx)]
# # # # # # #         except IndexError:
# # # # # # #             continue

# # # # # # #         curr_metadata = {
# # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # #         }

# # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # #             try:
# # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # #                 if response.status_code == 200:
# # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # #                 else:
# # # # # # #                     artwork_image = None
# # # # # # #             except:
# # # # # # #                 artwork_image = None
# # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # #             artwork_image = image_data_or_url
# # # # # # #         else:
# # # # # # #             artwork_image = None

# # # # # # #         if artwork_image:
# # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # #         else:
# # # # # # #             img_base64 = None

# # # # # # #         recommendations.append({
# # # # # # #             'index': idx,
# # # # # # #             'artist': curr_metadata['artist'],
# # # # # # #             'style': curr_metadata['style'],
# # # # # # #             'genre': curr_metadata['genre'],
# # # # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # # # #             'image': img_base64
# # # # # # #         })

# # # # # # #     return jsonify({'images': recommendations}), 200

# # # # # # # @app.route('/like-image', methods=['POST'])
# # # # # # # @token_required
# # # # # # # def like_image(current_user_email):
# # # # # # #     """
# # # # # # #     Records a user's like for an image and updates embeddings.
# # # # # # #     """
# # # # # # #     data = request.get_json()
# # # # # # #     image_index = data.get('image_index')

# # # # # # #     if image_index is None:
# # # # # # #         return jsonify({'message': 'Image index is required.'}), 400

# # # # # # #     # Record the like
# # # # # # #     try:
# # # # # # #         user_likes = load_user_likes()
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # # #     new_like = pd.DataFrame({
# # # # # # #         'UserEmail': [current_user_email],
# # # # # # #         'ImageIndex': [image_index],
# # # # # # #         'LikedAt': [datetime.datetime.utcnow()]
# # # # # # #     })

# # # # # # #     try:
# # # # # # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # # # # # #         save_user_likes(user_likes)
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # # # # # #     # Update user embedding after k likes
# # # # # # #     k = 5  # Define k as needed
# # # # # # #     user_total_likes = user_likes[user_likes['UserEmail'] == current_user_email].shape[0]

# # # # # # #     if user_total_likes % k == 0:
# # # # # # #         try:
# # # # # # #             user_embeddings = load_user_embeddings()
# # # # # # #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # #             if user_embedding_row.empty:
# # # # # # #                 # Initialize embedding with zeros if not found
# # # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # #                 zero_embedding = np.zeros(embedding_dim)
# # # # # # #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # #                 new_user_embedding = pd.DataFrame({
# # # # # # #                     'UserEmail': [current_user_email],
# # # # # # #                     'Embedding': [zero_embedding_encoded],
# # # # # # #                     'LastRecommendedIndex': [0],
# # # # # # #                     'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # # #                 })
# # # # # # #                 user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # # # # #                 save_user_embeddings(user_embeddings)
# # # # # # #                 user_embedding = zero_embedding
# # # # # # #                 print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # # # # # #             else:
# # # # # # #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # # # # # #             # Fetch user's liked image embeddings
# # # # # # #             liked_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()
# # # # # # #             if all_embeddings is not None and len(liked_indices) > 0:
# # # # # # #                 liked_embeddings = all_embeddings[liked_indices]
# # # # # # #                 # Compute the average of liked embeddings
# # # # # # #                 average_liked_embedding = np.mean(liked_embeddings, axis=0)
# # # # # # #                 average_liked_embedding = average_liked_embedding / np.linalg.norm(average_liked_embedding)
# # # # # # #             else:
# # # # # # #                 # If no embeddings available, use zero embedding
# # # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # #                 average_liked_embedding = np.zeros(embedding_dim)

# # # # # # #             # Combine with previous embedding
# # # # # # #             combined_embedding = combine_embeddings_for_recommendation(
# # # # # # #                 average_liked_embedding, user_embedding, weight=0.7
# # # # # # #             )
# # # # # # #             combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

# # # # # # #             # Update the embedding
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# # # # # # #             # Reset LastRecommendedIndex since embedding has been updated
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# # # # # # #             # Update LastEmbeddingUpdate timestamp
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # #             print(f"Updated embedding for user {current_user_email} after {k} likes.")
# # # # # # #         except Exception as e:
# # # # # # #             return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # # # # # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # # # # # @app.route('/recommend-images', methods=['GET'])
# # # # # # # @token_required
# # # # # # # def recommend_images(current_user_email):
# # # # # # #     """
# # # # # # #     Provides personalized recommendations based on user embeddings.
# # # # # # #     """
# # # # # # #     try:
# # # # # # #         user_embeddings = load_user_embeddings()
# # # # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # # #         if user_embedding_row.empty:
# # # # # # #             # Initialize embedding with zeros if not found
# # # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # # #             new_embedding = pd.DataFrame({
# # # # # # #                 'UserEmail': [current_user_email],
# # # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # # #                 'LastRecommendedIndex': [0],
# # # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # # #             })
# # # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # # #             save_user_embeddings(user_embeddings)
# # # # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # # # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # # # # # #         else:
# # # # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # # # #     # Check if user has liked any images
# # # # # # #     try:
# # # # # # #         user_likes = load_user_likes()
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # # # #     if not user_liked_images:
# # # # # # #         # User hasn't liked any images yet, return random 40 images
# # # # # # #         if train_data is not None:
# # # # # # #             num_images = len(train_data)
# # # # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # # # #         else:
# # # # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # # # #     else:
# # # # # # #         if all_embeddings is None:
# # # # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # # # #         # Ensure user_embedding has the correct dimension
# # # # # # #         embedding_dim = all_embeddings.shape[1]
# # # # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # # # #             else:
# # # # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # # #             # Update the embedding in the dataframe
# # # # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # # #             save_user_embeddings(user_embeddings)

# # # # # # #         # Compute similarities
# # # # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # # # #         # Exclude already liked images
# # # # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # # # #         # Fetch LastRecommendedIndex
# # # # # # #         try:
# # # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # # #         except:
# # # # # # #             last_recommended_index = 0

# # # # # # #         # Define batch size
# # # # # # #         batch_size = 10

# # # # # # #         # Select the next batch
# # # # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # # # #         # Update LastRecommendedIndex
# # # # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # # # #         save_user_embeddings(user_embeddings)

# # # # # # #     recommendations = []

# # # # # # #     for idx in indices:
# # # # # # #         try:
# # # # # # #             artwork = train_data[idx]
# # # # # # #         except IndexError:
# # # # # # #             continue

# # # # # # #         curr_metadata = {
# # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # #         }

# # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # #             try:
# # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # #                 if response.status_code == 200:
# # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # #                 else:
# # # # # # #                     artwork_image = None
# # # # # # #             except:
# # # # # # #                 artwork_image = None
# # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # #             artwork_image = image_data_or_url
# # # # # # #         else:
# # # # # # #             artwork_image = None

# # # # # # #         if artwork_image:
# # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # #         else:
# # # # # # #             img_base64 = None

# # # # # # #         recommendations.append({
# # # # # # #             'index': idx,
# # # # # # #             'artist': curr_metadata['artist'],
# # # # # # #             'style': curr_metadata['style'],
# # # # # # #             'genre': curr_metadata['genre'],
# # # # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # # # #             'image': img_base64
# # # # # # #         })

# # # # # # #     return jsonify({'recommendations': recommendations}), 200

# # # # # # # @app.route('/chat', methods=['POST'])
# # # # # # # @token_required
# # # # # # # def chat(current_user_email):
# # # # # # #     """
# # # # # # #     Handle chat requests with text and optional image.
# # # # # # #     Processes the inputs and returns a response.
# # # # # # #     """
# # # # # # #     text = request.form.get('text', '').strip()
# # # # # # #     image_file = request.files.get('image', None)

# # # # # # #     image_data = None
# # # # # # #     if image_file:
# # # # # # #         try:
# # # # # # #             image_bytes = image_file.read()
# # # # # # #             image = Image.open(io.BytesIO(image_bytes))
# # # # # # #             image = image.convert('RGB')
# # # # # # #             image_data = image
# # # # # # #         except Exception as e:
# # # # # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # # # # #     try:
# # # # # # #         result = predict(text, image_data)
# # # # # # #         return jsonify(result), 200
# # # # # # #     except Exception as e:
# # # # # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # # # # def predict(text, image_data=None):
# # # # # # #     """
# # # # # # #     Process the input text and image, generate recommendations,
# # # # # # #     and return them with explanations and metadata.
# # # # # # #     """
# # # # # # #     if not all([
# # # # # # #         all_embeddings is not None, 
# # # # # # #         train_data is not None, 
# # # # # # #         clip_model is not None, 
# # # # # # #         clip_processor is not None, 
# # # # # # #         blip_model is not None, 
# # # # # # #         blip_processor is not None
# # # # # # #     ]):
# # # # # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # # # # #     input_image = image_data
# # # # # # #     user_text = text

# # # # # # #     if input_image:
# # # # # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # # # # #         print(f"Generated image caption: {image_caption}")
# # # # # # #     else:
# # # # # # #         image_caption = ""

# # # # # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # # # # #     print(f"Context-aware text: {context_aware_text}")

# # # # # # #     if input_image:
# # # # # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # # # # #     else:
# # # # # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # # # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # # # # #     print("Preprocessed inputs for CLIP.")

# # # # # # #     with torch.no_grad():
# # # # # # #         if input_image:
# # # # # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # # # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # #             image_features_np = image_features.cpu().detach().numpy()
# # # # # # #         else:
# # # # # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # # # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # # # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # # # # #         text_features_np = text_features.cpu().detach().numpy()
# # # # # # #     print("Generated and normalized image and text features using CLIP.")

# # # # # # #     weight_img = 0.1
# # # # # # #     weight_text = 0.9

# # # # # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # # # # #     final_embedding = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # # # # #     print("Computed final combined embedding.")

# # # # # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # # # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # # # # #     embedding_dim = all_embeddings.shape[1]
# # # # # # #     if final_embedding.shape[1] != embedding_dim:
# # # # # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # # # # #         if final_embedding.shape[1] > embedding_dim:
# # # # # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # # # # #             print("Trimmed final_embedding.")
# # # # # # #         else:
# # # # # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # # # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # # # # #             final_embedding = np.hstack((final_embedding, padding))
# # # # # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # # # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # # # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # # # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # # # # #     top_n = 10
# # # # # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # # # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # # # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # # # # #     recommendations = []

# # # # # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # # # # #         try:
# # # # # # #             artwork = train_data[i]
# # # # # # #         except IndexError:
# # # # # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # # # # #             continue

# # # # # # #         curr_metadata = {
# # # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # # #         }

# # # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # # #         if isinstance(image_data_or_url, str):
# # # # # # #             try:
# # # # # # #                 response = requests.get(image_data_or_url)
# # # # # # #                 if response.status_code == 200:
# # # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # # #                 else:
# # # # # # #                     artwork_image = None
# # # # # # #             except Exception as e:
# # # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # # #                 artwork_image = None
# # # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # # #             artwork_image = image_data_or_url
# # # # # # #         else:
# # # # # # #             artwork_image = None

# # # # # # #         if artwork_image:
# # # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # # #         else:
# # # # # # #             img_base64 = None

# # # # # # #         recommendations.append({
# # # # # # #             'rank': rank,
# # # # # # #             'index': i,
# # # # # # #             'artist': curr_metadata['artist'],
# # # # # # #             'style': curr_metadata['style'],
# # # # # # #             'genre': curr_metadata['genre'],
# # # # # # #             # 'description': curr_metadata['description'],
# # # # # # #             'image': img_base64
# # # # # # #         })

# # # # # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # # # # #     return {
# # # # # # #         'response': response_text,
# # # # # # #         'recommendations': recommendations
# # # # # # #     }

# # # # # # # if __name__ == '__main__':
# # # # # # #     app.run(debug=True)


# # # # # # # backend/app.py

# # # # # # from flask import Flask, request, jsonify
# # # # # # import pandas as pd
# # # # # # import os
# # # # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # # # import jwt
# # # # # # import datetime
# # # # # # from functools import wraps
# # # # # # from flask_cors import CORS
# # # # # # from dotenv import load_dotenv
# # # # # # import io
# # # # # # from PIL import Image
# # # # # # import numpy as np
# # # # # # import torch
# # # # # # from transformers import CLIPProcessor, CLIPModel
# # # # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # # # from datasets import load_dataset
# # # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # # import requests
# # # # # # import base64
# # # # # # import json
# # # # # # from filelock import FileLock, Timeout

# # # # # # def display_image(image_data):
# # # # # #     # Function to display images (not used in backend)
# # # # # #     pass

# # # # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # # # #     with torch.no_grad():
# # # # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # # # #     return caption

# # # # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # # # #     margin = 0.05
# # # # # #     if sim_image > sim_text + margin:
# # # # # #         reason = "the style and composition of the input image."
# # # # # #     elif sim_text > sim_image + margin:
# # # # # #         reason = "your textual preferences for nature and the specified colors."
# # # # # #     else:
# # # # # #         reason = "a balanced combination of both your image and textual preferences."

# # # # # #     explanation = (
# # # # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # # # #         f"is recommended {reason} "
# # # # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # # # #     )
# # # # # #     return explanation

# # # # # # def encode_image_to_base64(image):
# # # # # #     buffered = io.BytesIO()
# # # # # #     image.save(buffered, format="JPEG")
# # # # # #     img_bytes = buffered.getvalue()
# # # # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # # # #     return img_base64

# # # # # # def decode_embedding(embedding_str):
# # # # # #     return np.array(json.loads(embedding_str))

# # # # # # def encode_embedding(embedding_array):
# # # # # #     return json.dumps(embedding_array.tolist())

# # # # # # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# # # # # #     """
# # # # # #     Combines the current embedding with the previous one using a weighted average.
# # # # # #     """
# # # # # #     if previous_embedding is None:
# # # # # #         return current_embedding
# # # # # #     return weight * current_embedding + (1 - weight) * previous_embedding

# # # # # # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# # # # # #     """
# # # # # #     Recommends the top-k similar artworks based on cosine similarity.
# # # # # #     """
# # # # # #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# # # # # #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# # # # # #     return top_k_indices

# # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # print(f"Using device: {device}")

# # # # # # # Load combined embeddings
# # # # # # try:
# # # # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # # # except FileNotFoundError:
# # # # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # # # #     all_embeddings = None

# # # # # # if all_embeddings is not None:
# # # # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # # # else:
# # # # # #     print("Skipping normalization due to missing embeddings.")

# # # # # # # Load the dataset (e.g., WikiArt for training data)
# # # # # # try:
# # # # # #     ds = load_dataset("Artificio/WikiArt")
# # # # # #     train_data = ds['train']
# # # # # #     print("Loaded WikiArt dataset successfully.")
# # # # # # except Exception as e:
# # # # # #     print(f"Error loading dataset: {e}")
# # # # # #     train_data = None

# # # # # # # Load CLIP model and processor
# # # # # # try:
# # # # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # # # #     clip_model.to(device)
# # # # # #     print("Loaded CLIP model and processor successfully.")
# # # # # # except Exception as e:
# # # # # #     print(f"Error loading CLIP model: {e}")
# # # # # #     clip_model = None
# # # # # #     clip_processor = None

# # # # # # # Load BLIP model and processor for image captioning
# # # # # # try:
# # # # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # # #     blip_model.to(device)
# # # # # #     print("Loaded BLIP model and processor successfully.")
# # # # # # except Exception as e:
# # # # # #     print(f"Error loading BLIP model: {e}")
# # # # # #     blip_model = None
# # # # # #     blip_processor = None

# # # # # # # Load environment variables from .env file
# # # # # # load_dotenv()

# # # # # # app = Flask(__name__)
# # # # # # CORS(app)

# # # # # # # Retrieve the secret key from environment variables
# # # # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # # # Ensure that the secret key is set
# # # # # # if not app.config['SECRET_KEY']:
# # # # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # # # Define Excel database files
# # # # # # DATABASE_USERS = 'users.xlsx'
# # # # # # DATABASE_LIKES = 'user_likes.xlsx'
# # # # # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'

# # # # # # # Initialize Excel files if they don't exist
# # # # # # for db_file, columns in [
# # # # # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # # # # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # # # # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # # # ]:
# # # # # #     if not os.path.exists(db_file):
# # # # # #         df = pd.DataFrame(columns=columns)
# # # # # #         df.to_excel(db_file, index=False, engine='openpyxl')
# # # # # #         print(f"Created {db_file} with columns: {columns}")

# # # # # # def load_users():
# # # # # #     lock_path = DATABASE_USERS + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             return pd.read_excel(DATABASE_USERS, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for loading users.")
# # # # # #     except Exception as e:
# # # # # #         # Handle decompression errors if any
# # # # # #         if 'decompressing data' in str(e):
# # # # # #             print(f"Corrupted {DATABASE_USERS}. Recreating it.")
# # # # # #             df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # # # # #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # # # #             return df
# # # # # #         else:
# # # # # #             raise e

# # # # # # def save_users(df):
# # # # # #     lock_path = DATABASE_USERS + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for saving users.")
# # # # # #     except Exception as e:
# # # # # #         raise e

# # # # # # def load_user_likes():
# # # # # #     lock_path = DATABASE_LIKES + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             return pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for loading user likes.")
# # # # # #     except Exception as e:
# # # # # #         # Handle decompression errors
# # # # # #         if 'decompressing data' in str(e):
# # # # # #             print(f"Corrupted {DATABASE_LIKES}. Recreating it.")
# # # # # #             df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# # # # # #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # # # #             return df
# # # # # #         else:
# # # # # #             raise e

# # # # # # def save_user_likes(df):
# # # # # #     lock_path = DATABASE_LIKES + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for saving user likes.")
# # # # # #     except Exception as e:
# # # # # #         raise e

# # # # # # def load_user_embeddings():
# # # # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             return pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for loading user embeddings.")
# # # # # #     except Exception as e:
# # # # # #         # Handle decompression errors
# # # # # #         if 'decompressing data' in str(e):
# # # # # #             print(f"Corrupted {DATABASE_EMBEDDINGS}. Recreating it.")
# # # # # #             df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # # #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # # # #             return df
# # # # # #         else:
# # # # # #             raise e

# # # # # # def save_user_embeddings(df):
# # # # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # # # #     lock = FileLock(lock_path)
# # # # # #     try:
# # # # # #         with lock.acquire(timeout=10):
# # # # # #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # # # #     except Timeout:
# # # # # #         raise Exception("Could not acquire lock for saving user embeddings.")
# # # # # #     except Exception as e:
# # # # # #         raise e

# # # # # # def token_required(f):
# # # # # #     @wraps(f)
# # # # # #     def decorated(*args, **kwargs):
# # # # # #         token = None

# # # # # #         if 'Authorization' in request.headers:
# # # # # #             auth_header = request.headers['Authorization']
# # # # # #             try:
# # # # # #                 token = auth_header.split(" ")[1]
# # # # # #             except IndexError:
# # # # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # # # #         if not token:
# # # # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # # # #         try:
# # # # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # # # #             current_user_email = data['email']
# # # # # #         except jwt.ExpiredSignatureError:
# # # # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # # # #         except jwt.InvalidTokenError:
# # # # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # # # #         try:
# # # # # #             users = load_users()
# # # # # #         except Exception as e:
# # # # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # #         user = users[users['Email'] == current_user_email]
# # # # # #         if user.empty:
# # # # # #             return jsonify({'message': 'User not found!'}), 401

# # # # # #         return f(current_user_email, *args, **kwargs)

# # # # # #     return decorated

# # # # # # @app.route('/signup', methods=['POST'])
# # # # # # def signup():
# # # # # #     data = request.get_json()
# # # # # #     full_name = data.get('full_name')
# # # # # #     email = data.get('email')
# # # # # #     password = data.get('password')

# # # # # #     if not all([full_name, email, password]):
# # # # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # # # #     try:
# # # # # #         users = load_users()
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # #     if email in users['Email'].values:
# # # # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # # # #     hashed_password = generate_password_hash(password)

# # # # # #     new_user = pd.DataFrame({
# # # # # #         'FullName': [full_name],
# # # # # #         'Email': [email],
# # # # # #         'Password': [hashed_password]
# # # # # #     })

# # # # # #     try:
# # # # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # # # #     try:
# # # # # #         save_users(users)
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # # # #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# # # # # #     try:
# # # # # #         user_embeddings = load_user_embeddings()
# # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # #             new_embedding = pd.DataFrame({
# # # # # #                 'UserEmail': [email],
# # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # #                 'LastRecommendedIndex': [0],
# # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # #             })
# # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # #             save_user_embeddings(user_embeddings)
# # # # # #             print(f"Initialized zero embedding for user {email}.")
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # # # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # # # @app.route('/login', methods=['POST'])
# # # # # # def login():
# # # # # #     data = request.get_json()
# # # # # #     email = data.get('email')
# # # # # #     password = data.get('password')

# # # # # #     if not all([email, password]):
# # # # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # # # #     try:
# # # # # #         users = load_users()
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # # #     user = users[users['Email'] == email]

# # # # # #     if user.empty:
# # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # #     stored_password = user.iloc[0]['Password']
# # # # # #     full_name = user.iloc[0]['FullName']

# # # # # #     if not check_password_hash(stored_password, password):
# # # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # # #     try:
# # # # # #         token = jwt.encode({
# # # # # #             'email': email,
# # # # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # # # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # # # #     # Ensure user has an embedding; initialize if not
# # # # # #     try:
# # # # # #         user_embeddings = load_user_embeddings()
# # # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # #             new_embedding = pd.DataFrame({
# # # # # #                 'UserEmail': [email],
# # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # #                 'LastRecommendedIndex': [0],
# # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # #             })
# # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # #             save_user_embeddings(user_embeddings)
# # # # # #             print(f"Initialized zero embedding for user {email} on login.")
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # # # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # # # @app.route('/protected', methods=['GET'])
# # # # # # @token_required
# # # # # # def protected_route(current_user_email):
# # # # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # # # --- New Endpoints ---

# # # # # # @app.route('/get-images', methods=['GET'])
# # # # # # @token_required
# # # # # # def get_images(current_user_email):
# # # # # #     """
# # # # # #     Fetch a batch of images for the user.
# # # # # #     If the user has not liked any images, return random indices.
# # # # # #     Otherwise, fetch based on recommendations.
# # # # # #     """
# # # # # #     try:
# # # # # #         user_likes = load_user_likes()
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # # #     try:
# # # # # #         user_embeddings = load_user_embeddings()
# # # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # #         if user_embedding_row.empty:
# # # # # #             # Initialize embedding with zeros if not found
# # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # #             new_embedding = pd.DataFrame({
# # # # # #                 'UserEmail': [current_user_email],
# # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # #                 'LastRecommendedIndex': [0],
# # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # #             })
# # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # #             save_user_embeddings(user_embeddings)
# # # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # # #             print(f"Initialized zero embedding for user {current_user_email} in /get-images.")
# # # # # #         else:
# # # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # # #     # Check if user has liked any images
# # # # # #     if not user_liked_images:
# # # # # #         # User hasn't liked any images yet, return random 40 images
# # # # # #         if train_data is not None:
# # # # # #             num_images = len(train_data)
# # # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # # #         else:
# # # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # # #     else:
# # # # # #         if all_embeddings is None:
# # # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # # #         # Ensure user_embedding has the correct dimension
# # # # # #         embedding_dim = all_embeddings.shape[1]
# # # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # # #             else:
# # # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # #             # Update the embedding in the dataframe
# # # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # # # # #             save_user_embeddings(user_embeddings)

# # # # # #         # Compute similarities
# # # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # # #         # Exclude already liked images
# # # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # # #         # Fetch LastRecommendedIndex
# # # # # #         try:
# # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # #         except:
# # # # # #             last_recommended_index = 0

# # # # # #         # Define batch size
# # # # # #         batch_size = 10

# # # # # #         # Select the next batch
# # # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # # #         # Update LastRecommendedIndex
# # # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # # #         save_user_embeddings(user_embeddings)

# # # # # #     recommendations = []

# # # # # #     for idx in indices:
# # # # # #         try:
# # # # # #             artwork = train_data[int(idx)]
# # # # # #         except IndexError:
# # # # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # # # #             continue
# # # # # #         except TypeError as te:
# # # # # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # # # #             continue

# # # # # #         curr_metadata = {
# # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # #         }

# # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # #         if isinstance(image_data_or_url, str):
# # # # # #             try:
# # # # # #                 response = requests.get(image_data_or_url)
# # # # # #                 if response.status_code == 200:
# # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # #                 else:
# # # # # #                     artwork_image = None
# # # # # #             except Exception as e:
# # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # #                 artwork_image = None
# # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # #             artwork_image = image_data_or_url
# # # # # #         else:
# # # # # #             artwork_image = None

# # # # # #         if artwork_image:
# # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # #         else:
# # # # # #             img_base64 = None

# # # # # #         recommendations.append({
# # # # # #             'index': idx,
# # # # # #             'artist': curr_metadata['artist'],
# # # # # #             'style': curr_metadata['style'],
# # # # # #             'genre': curr_metadata['genre'],
# # # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # # #             'image': img_base64
# # # # # #         })

# # # # # #     return jsonify({'images': recommendations}), 200

# # # # # # @app.route('/like-image', methods=['POST'])
# # # # # # @token_required
# # # # # # def like_image(current_user_email):
# # # # # #     """
# # # # # #     Records a user's like for an image and updates embeddings.
# # # # # #     """
# # # # # #     data = request.get_json()
# # # # # #     image_index = data.get('image_index')

# # # # # #     if image_index is None:
# # # # # #         return jsonify({'message': 'Image index is required.'}), 400

# # # # # #     # Record the like
# # # # # #     try:
# # # # # #         user_likes = load_user_likes()
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # #     new_like = pd.DataFrame({
# # # # # #         'UserEmail': [current_user_email],
# # # # # #         'ImageIndex': [image_index],
# # # # # #         'LikedAt': [datetime.datetime.utcnow()]
# # # # # #     })

# # # # # #     try:
# # # # # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # # # # #         save_user_likes(user_likes)
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # # # # #     # Update user embedding after k likes
# # # # # #     k = 5  # Define k as needed
# # # # # #     user_total_likes = user_likes[user_likes['UserEmail'] == current_user_email].shape[0]

# # # # # #     if user_total_likes % k == 0:
# # # # # #         try:
# # # # # #             user_embeddings = load_user_embeddings()
# # # # # #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # #             if user_embedding_row.empty:
# # # # # #                 # Initialize embedding with zeros if not found
# # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # #                 zero_embedding = np.zeros(embedding_dim)
# # # # # #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # #                 new_user_embedding = pd.DataFrame({
# # # # # #                     'UserEmail': [current_user_email],
# # # # # #                     'Embedding': [zero_embedding_encoded],
# # # # # #                     'LastRecommendedIndex': [0],
# # # # # #                     'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # #                 })
# # # # # #                 user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # # # #                 save_user_embeddings(user_embeddings)
# # # # # #                 user_embedding = zero_embedding
# # # # # #                 print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # # # # #             else:
# # # # # #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # # # # #             # Fetch user's liked image embeddings
# # # # # #             liked_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()
# # # # # #             if all_embeddings is not None and len(liked_indices) > 0:
# # # # # #                 # Ensure indices are within range
# # # # # #                 liked_indices = [i for i in liked_indices if 0 <= i < all_embeddings.shape[0]]
# # # # # #                 if len(liked_indices) > 0:
# # # # # #                     liked_embeddings = all_embeddings[liked_indices]
# # # # # #                     # Compute the average of liked embeddings
# # # # # #                     average_liked_embedding = np.mean(liked_embeddings, axis=0)
# # # # # #                     if np.linalg.norm(average_liked_embedding) != 0:
# # # # # #                         average_liked_embedding = average_liked_embedding / np.linalg.norm(average_liked_embedding)
# # # # # #                 else:
# # # # # #                     # If no valid liked_indices, use zero embedding
# # # # # #                     embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # #                     average_liked_embedding = np.zeros(embedding_dim)
# # # # # #             else:
# # # # # #                 # If no embeddings available, use zero embedding
# # # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # #                 average_liked_embedding = np.zeros(embedding_dim)

# # # # # #             # Combine with previous embedding
# # # # # #             combined_embedding = combine_embeddings_for_recommendation(
# # # # # #                 average_liked_embedding, user_embedding, weight=0.7
# # # # # #             )
# # # # # #             norm = np.linalg.norm(combined_embedding)
# # # # # #             if norm != 0:
# # # # # #                 combined_embedding = combined_embedding / norm
# # # # # #             else:
# # # # # #                 combined_embedding = combined_embedding

# # # # # #             # Update the embedding
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# # # # # #             # Reset LastRecommendedIndex since embedding has been updated
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# # # # # #             # Update LastEmbeddingUpdate timestamp
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# # # # # #             save_user_embeddings(user_embeddings)
# # # # # #             print(f"Updated embedding for user {current_user_email} after {k} likes.")
# # # # # #         except Exception as e:
# # # # # #             return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # # # # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # # # # @app.route('/recommend-images', methods=['GET'])
# # # # # # @token_required
# # # # # # def recommend_images(current_user_email):
# # # # # #     """
# # # # # #     Provides personalized recommendations based on user embeddings.
# # # # # #     """
# # # # # #     try:
# # # # # #         user_embeddings = load_user_embeddings()
# # # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # # #         if user_embedding_row.empty:
# # # # # #             # Initialize embedding with zeros if not found
# # # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # # #             new_embedding = pd.DataFrame({
# # # # # #                 'UserEmail': [current_user_email],
# # # # # #                 'Embedding': [zero_embedding_encoded],
# # # # # #                 'LastRecommendedIndex': [0],
# # # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # # #             })
# # # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # # #             save_user_embeddings(user_embeddings)
# # # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # # # # #         else:
# # # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # # #     # Check if user has liked any images
# # # # # #     try:
# # # # # #         user_likes = load_user_likes()
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # # #     if not user_liked_images:
# # # # # #         # User hasn't liked any images yet, return random 40 images
# # # # # #         if train_data is not None:
# # # # # #             num_images = len(train_data)
# # # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # # #         else:
# # # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # # #     else:
# # # # # #         if all_embeddings is None:
# # # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # # #         # Ensure user_embedding has the correct dimension
# # # # # #         embedding_dim = all_embeddings.shape[1]
# # # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # # #             else:
# # # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # # #             # Update the embedding in the dataframe
# # # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # # #             save_user_embeddings(user_embeddings)

# # # # # #         # Compute similarities
# # # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # # #         # Exclude already liked images
# # # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # # #         # Fetch LastRecommendedIndex
# # # # # #         try:
# # # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # # #         except:
# # # # # #             last_recommended_index = 0

# # # # # #         # Define batch size
# # # # # #         batch_size = 10

# # # # # #         # Select the next batch
# # # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # # #         # Update LastRecommendedIndex
# # # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # # #         save_user_embeddings(user_embeddings)

# # # # # #     recommendations = []

# # # # # #     for idx in indices:
# # # # # #         try:
# # # # # #             artwork = train_data[int(idx)]
# # # # # #         except IndexError:
# # # # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # # # #             continue

# # # # # #         curr_metadata = {
# # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # #         }

# # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # #         if isinstance(image_data_or_url, str):
# # # # # #             try:
# # # # # #                 response = requests.get(image_data_or_url)
# # # # # #                 if response.status_code == 200:
# # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # #                 else:
# # # # # #                     artwork_image = None
# # # # # #             except Exception as e:
# # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # #                 artwork_image = None
# # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # #             artwork_image = image_data_or_url
# # # # # #         else:
# # # # # #             artwork_image = None

# # # # # #         if artwork_image:
# # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # #         else:
# # # # # #             img_base64 = None

# # # # # #         recommendations.append({
# # # # # #             'index': idx,
# # # # # #             'artist': curr_metadata['artist'],
# # # # # #             'style': curr_metadata['style'],
# # # # # #             'genre': curr_metadata['genre'],
# # # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # # #             'image': img_base64
# # # # # #         })

# # # # # #     return jsonify({'recommendations': recommendations}), 200

# # # # # # @app.route('/chat', methods=['POST'])
# # # # # # @token_required
# # # # # # def chat(current_user_email):
# # # # # #     """
# # # # # #     Handle chat requests with text and optional image.
# # # # # #     Processes the inputs and returns a response.
# # # # # #     """
# # # # # #     text = request.form.get('text', '').strip()
# # # # # #     image_file = request.files.get('image', None)

# # # # # #     image_data = None
# # # # # #     if image_file:
# # # # # #         try:
# # # # # #             image_bytes = image_file.read()
# # # # # #             image = Image.open(io.BytesIO(image_bytes))
# # # # # #             image = image.convert('RGB')
# # # # # #             image_data = image
# # # # # #         except Exception as e:
# # # # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # # # #     try:
# # # # # #         result = predict(text, image_data)
# # # # # #         return jsonify(result), 200
# # # # # #     except Exception as e:
# # # # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # # # def predict(text, image_data=None):
# # # # # #     """
# # # # # #     Process the input text and image, generate recommendations,
# # # # # #     and return them with explanations and metadata.
# # # # # #     """
# # # # # #     if not all([
# # # # # #         all_embeddings is not None, 
# # # # # #         train_data is not None, 
# # # # # #         clip_model is not None, 
# # # # # #         clip_processor is not None, 
# # # # # #         blip_model is not None, 
# # # # # #         blip_processor is not None
# # # # # #     ]):
# # # # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # # # #     input_image = image_data
# # # # # #     user_text = text

# # # # # #     if input_image:
# # # # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # # # #         print(f"Generated image caption: {image_caption}")
# # # # # #     else:
# # # # # #         image_caption = ""

# # # # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # # # #     print(f"Context-aware text: {context_aware_text}")

# # # # # #     if input_image:
# # # # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # # # #     else:
# # # # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # # # #     print("Preprocessed inputs for CLIP.")

# # # # # #     with torch.no_grad():
# # # # # #         if input_image:
# # # # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # # # #             image_features_np = image_features.cpu().detach().numpy()
# # # # # #         else:
# # # # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # # # #         text_features_np = text_features.cpu().detach().numpy()
# # # # # #     print("Generated and normalized image and text features using CLIP.")

# # # # # #     weight_img = 0.1
# # # # # #     weight_text = 0.9

# # # # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # # # #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # # # #     if norm != 0:
# # # # # #         final_embedding = final_embedding / norm
# # # # # #     else:
# # # # # #         final_embedding = final_embedding
# # # # # #     print("Computed final combined embedding.")

# # # # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # # # #     embedding_dim = all_embeddings.shape[1]
# # # # # #     if final_embedding.shape[1] != embedding_dim:
# # # # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # # # #         if final_embedding.shape[1] > embedding_dim:
# # # # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # # # #             print("Trimmed final_embedding.")
# # # # # #         else:
# # # # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # # # #             final_embedding = np.hstack((final_embedding, padding))
# # # # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # # # #     top_n = 10
# # # # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # # # #     recommendations = []

# # # # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # # # #         try:
# # # # # #             artwork = train_data[int(i)]
# # # # # #         except IndexError:
# # # # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # # # #             continue

# # # # # #         curr_metadata = {
# # # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # # #             "description": artwork.get('description', 'No Description Available')
# # # # # #         }

# # # # # #         image_data_or_url = artwork.get('image', None)

# # # # # #         if isinstance(image_data_or_url, str):
# # # # # #             try:
# # # # # #                 response = requests.get(image_data_or_url)
# # # # # #                 if response.status_code == 200:
# # # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # # #                 else:
# # # # # #                     artwork_image = None
# # # # # #             except Exception as e:
# # # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # # #                 artwork_image = None
# # # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # # #             artwork_image = image_data_or_url
# # # # # #         else:
# # # # # #             artwork_image = None

# # # # # #         if artwork_image:
# # # # # #             img_base64 = encode_image_to_base64(artwork_image)
# # # # # #         else:
# # # # # #             img_base64 = None

# # # # # #         recommendations.append({
# # # # # #             'rank': rank,
# # # # # #             'index': i,
# # # # # #             'artist': curr_metadata['artist'],
# # # # # #             'style': curr_metadata['style'],
# # # # # #             'genre': curr_metadata['genre'],
# # # # # #             # 'description': curr_metadata['description'],
# # # # # #             'image': img_base64
# # # # # #         })

# # # # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # # # #     return {
# # # # # #         'response': response_text,
# # # # # #         'recommendations': recommendations
# # # # # #     }

# # # # # # if __name__ == '__main__':
# # # # # #     app.run(debug=True)


# # # # # # backend/app.py

# # # # # from flask import Flask, request, jsonify
# # # # # import pandas as pd
# # # # # import os
# # # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # # import jwt
# # # # # import datetime
# # # # # from functools import wraps
# # # # # from flask_cors import CORS
# # # # # from dotenv import load_dotenv
# # # # # import io
# # # # # from PIL import Image
# # # # # import numpy as np
# # # # # import torch
# # # # # from transformers import CLIPProcessor, CLIPModel
# # # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # # from datasets import load_dataset
# # # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # # import requests
# # # # # import base64
# # # # # import json
# # # # # from filelock import FileLock, Timeout

# # # # # def display_image(image_data):
# # # # #     # Function to display images (not used in backend)
# # # # #     pass

# # # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # # #     with torch.no_grad():
# # # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # # #     return caption

# # # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # # #     margin = 0.05
# # # # #     if sim_image > sim_text + margin:
# # # # #         reason = "the style and composition of the input image."
# # # # #     elif sim_text > sim_image + margin:
# # # # #         reason = "your textual preferences for nature and the specified colors."
# # # # #     else:
# # # # #         reason = "a balanced combination of both your image and textual preferences."

# # # # #     explanation = (
# # # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # # #         f"is recommended {reason} "
# # # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # # #     )
# # # # #     return explanation

# # # # # def encode_image_to_base64(image):
# # # # #     buffered = io.BytesIO()
# # # # #     image.save(buffered, format="JPEG")
# # # # #     img_bytes = buffered.getvalue()
# # # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # # #     return img_base64

# # # # # def decode_embedding(embedding_str):
# # # # #     return np.array(json.loads(embedding_str))

# # # # # def encode_embedding(embedding_array):
# # # # #     return json.dumps(embedding_array.tolist())

# # # # # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# # # # #     """
# # # # #     Combines the current embedding with the previous one using a weighted average.
# # # # #     """
# # # # #     if previous_embedding is None:
# # # # #         return current_embedding
# # # # #     return weight * current_embedding + (1 - weight) * previous_embedding

# # # # # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# # # # #     """
# # # # #     Recommends the top-k similar artworks based on cosine similarity.
# # # # #     """
# # # # #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# # # # #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# # # # #     return top_k_indices

# # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # print(f"Using device: {device}")

# # # # # # Load combined embeddings
# # # # # try:
# # # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # # except FileNotFoundError:
# # # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # # #     all_embeddings = None

# # # # # if all_embeddings is not None:
# # # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # # else:
# # # # #     print("Skipping normalization due to missing embeddings.")

# # # # # # Load the dataset (e.g., WikiArt for training data)
# # # # # try:
# # # # #     ds = load_dataset("Artificio/WikiArt")
# # # # #     train_data = ds['train']
# # # # #     print("Loaded WikiArt dataset successfully.")
# # # # # except Exception as e:
# # # # #     print(f"Error loading dataset: {e}")
# # # # #     train_data = None

# # # # # # Load CLIP model and processor
# # # # # try:
# # # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # # #     clip_model.to(device)
# # # # #     print("Loaded CLIP model and processor successfully.")
# # # # # except Exception as e:
# # # # #     print(f"Error loading CLIP model: {e}")
# # # # #     clip_model = None
# # # # #     clip_processor = None

# # # # # # Load BLIP model and processor for image captioning
# # # # # try:
# # # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # # #     blip_model.to(device)
# # # # #     print("Loaded BLIP model and processor successfully.")
# # # # # except Exception as e:
# # # # #     print(f"Error loading BLIP model: {e}")
# # # # #     blip_model = None
# # # # #     blip_processor = None

# # # # # # Load environment variables from .env file
# # # # # load_dotenv()

# # # # # app = Flask(__name__)
# # # # # CORS(app)

# # # # # # Retrieve the secret key from environment variables
# # # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # # Ensure that the secret key is set
# # # # # if not app.config['SECRET_KEY']:
# # # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # # Define Excel database files
# # # # # DATABASE_USERS = 'users.xlsx'
# # # # # DATABASE_LIKES = 'user_likes.xlsx'
# # # # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'

# # # # # # Initialize Excel files if they don't exist
# # # # # for db_file, columns in [
# # # # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # # # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # # # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # # ]:
# # # # #     if not os.path.exists(db_file):
# # # # #         try:
# # # # #             df = pd.DataFrame(columns=columns)
# # # # #             df.to_excel(db_file, index=False, engine='openpyxl')
# # # # #             print(f"Created {db_file} with columns: {columns}")
# # # # #         except Exception as e:
# # # # #             print(f"Error creating {db_file}: {e}")

# # # # # def load_users():
# # # # #     lock_path = DATABASE_USERS + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             return pd.read_excel(DATABASE_USERS, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for loading users.")
# # # # #     except Exception as e:
# # # # #         # Handle decompression errors or other read errors
# # # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # # #             print(f"Corrupted {DATABASE_USERS}. Recreating it.")
# # # # #             df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # # # #             try:
# # # # #                 df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # # #                 print(f"Recreated {DATABASE_USERS}.")
# # # # #             except Exception as ex:
# # # # #                 print(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # # # #                 raise Exception(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # # # #             return df
# # # # #         else:
# # # # #             raise e

# # # # # def save_users(df):
# # # # #     lock_path = DATABASE_USERS + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for saving users.")
# # # # #     except Exception as e:
# # # # #         raise e

# # # # # def load_user_likes():
# # # # #     lock_path = DATABASE_LIKES + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             return pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for loading user likes.")
# # # # #     except Exception as e:
# # # # #         # Handle decompression errors
# # # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # # #             print(f"Corrupted {DATABASE_LIKES}. Recreating it.")
# # # # #             df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# # # # #             try:
# # # # #                 df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # # #                 print(f"Recreated {DATABASE_LIKES}.")
# # # # #             except Exception as ex:
# # # # #                 print(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # # # #                 raise Exception(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # # # #             return df
# # # # #         else:
# # # # #             raise e

# # # # # def save_user_likes(df):
# # # # #     lock_path = DATABASE_LIKES + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for saving user likes.")
# # # # #     except Exception as e:
# # # # #         raise e

# # # # # def load_user_embeddings():
# # # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             return pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for loading user embeddings.")
# # # # #     except Exception as e:
# # # # #         # Handle decompression errors
# # # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # # #             print(f"Corrupted {DATABASE_EMBEDDINGS}. Recreating it.")
# # # # #             df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # #             try:
# # # # #                 df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # # #                 print(f"Recreated {DATABASE_EMBEDDINGS}.")
# # # # #             except Exception as ex:
# # # # #                 print(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # # # #                 raise Exception(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # # # #             return df
# # # # #         else:
# # # # #             raise e

# # # # # def save_user_embeddings(df):
# # # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # # #     lock = FileLock(lock_path)
# # # # #     try:
# # # # #         with lock.acquire(timeout=10):
# # # # #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # # #     except Timeout:
# # # # #         raise Exception("Could not acquire lock for saving user embeddings.")
# # # # #     except Exception as e:
# # # # #         raise e

# # # # # def token_required(f):
# # # # #     @wraps(f)
# # # # #     def decorated(*args, **kwargs):
# # # # #         token = None

# # # # #         if 'Authorization' in request.headers:
# # # # #             auth_header = request.headers['Authorization']
# # # # #             try:
# # # # #                 token = auth_header.split(" ")[1]
# # # # #             except IndexError:
# # # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # # #         if not token:
# # # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # # #         try:
# # # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # # #             current_user_email = data['email']
# # # # #         except jwt.ExpiredSignatureError:
# # # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # # #         except jwt.InvalidTokenError:
# # # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # # #         try:
# # # # #             users = load_users()
# # # # #         except Exception as e:
# # # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # #         user = users[users['Email'] == current_user_email]
# # # # #         if user.empty:
# # # # #             return jsonify({'message': 'User not found!'}), 401

# # # # #         return f(current_user_email, *args, **kwargs)

# # # # #     return decorated

# # # # # @app.route('/signup', methods=['POST'])
# # # # # def signup():
# # # # #     data = request.get_json()
# # # # #     full_name = data.get('full_name')
# # # # #     email = data.get('email')
# # # # #     password = data.get('password')

# # # # #     if not all([full_name, email, password]):
# # # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # # #     try:
# # # # #         users = load_users()
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # #     if email in users['Email'].values:
# # # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # # #     hashed_password = generate_password_hash(password)

# # # # #     new_user = pd.DataFrame({
# # # # #         'FullName': [full_name],
# # # # #         'Email': [email],
# # # # #         'Password': [hashed_password]
# # # # #     })

# # # # #     try:
# # # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # # #     try:
# # # # #         save_users(users)
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # # #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# # # # #     try:
# # # # #         user_embeddings = load_user_embeddings()
# # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # #             new_embedding = pd.DataFrame({
# # # # #                 'UserEmail': [email],
# # # # #                 'Embedding': [zero_embedding_encoded],
# # # # #                 'LastRecommendedIndex': [0],
# # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # #             })
# # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # #             save_user_embeddings(user_embeddings)
# # # # #             print(f"Initialized zero embedding for user {email}.")
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # # @app.route('/login', methods=['POST'])
# # # # # def login():
# # # # #     data = request.get_json()
# # # # #     email = data.get('email')
# # # # #     password = data.get('password')

# # # # #     if not all([email, password]):
# # # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # # #     try:
# # # # #         users = load_users()
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # # #     user = users[users['Email'] == email]

# # # # #     if user.empty:
# # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # #     stored_password = user.iloc[0]['Password']
# # # # #     full_name = user.iloc[0]['FullName']

# # # # #     if not check_password_hash(stored_password, password):
# # # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # # #     try:
# # # # #         token = jwt.encode({
# # # # #             'email': email,
# # # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # # #     # Ensure user has an embedding; initialize if not
# # # # #     try:
# # # # #         user_embeddings = load_user_embeddings()
# # # # #         if email not in user_embeddings['UserEmail'].values:
# # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # #             new_embedding = pd.DataFrame({
# # # # #                 'UserEmail': [email],
# # # # #                 'Embedding': [zero_embedding_encoded],
# # # # #                 'LastRecommendedIndex': [0],
# # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # #             })
# # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # #             save_user_embeddings(user_embeddings)
# # # # #             print(f"Initialized zero embedding for user {email} on login.")
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # # @app.route('/protected', methods=['GET'])
# # # # # @token_required
# # # # # def protected_route(current_user_email):
# # # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # # --- New Endpoints ---

# # # # # @app.route('/get-images', methods=['GET'])
# # # # # @token_required
# # # # # def get_images(current_user_email):
# # # # #     """
# # # # #     Fetch a batch of images for the user.
# # # # #     If the user has not liked any images, return random indices.
# # # # #     Otherwise, fetch based on recommendations.
# # # # #     """
# # # # #     try:
# # # # #         user_likes = load_user_likes()
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # #     try:
# # # # #         user_embeddings = load_user_embeddings()
# # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # #         if user_embedding_row.empty:
# # # # #             # Initialize embedding with zeros if not found
# # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # #             new_embedding = pd.DataFrame({
# # # # #                 'UserEmail': [current_user_email],
# # # # #                 'Embedding': [zero_embedding_encoded],
# # # # #                 'LastRecommendedIndex': [0],
# # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # #             })
# # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # #             save_user_embeddings(user_embeddings)
# # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # #             print(f"Initialized zero embedding for user {current_user_email} in /get-images.")
# # # # #         else:
# # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # #     # Check if user has liked any images
# # # # #     if not user_liked_images:
# # # # #         # User hasn't liked any images yet, return random 40 images
# # # # #         if train_data is not None:
# # # # #             num_images = len(train_data)
# # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # #         else:
# # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # #     else:
# # # # #         if all_embeddings is None:
# # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # #         # Ensure user_embedding has the correct dimension
# # # # #         embedding_dim = all_embeddings.shape[1]
# # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # #             else:
# # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # #             # Update the embedding in the dataframe
# # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # # # #             save_user_embeddings(user_embeddings)

# # # # #         # Compute similarities
# # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # #         # Exclude already liked images
# # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # #         # Fetch LastRecommendedIndex
# # # # #         try:
# # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # #         except:
# # # # #             last_recommended_index = 0

# # # # #         # Define batch size
# # # # #         batch_size = 10

# # # # #         # Select the next batch
# # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # #         # Update LastRecommendedIndex
# # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # #         save_user_embeddings(user_embeddings)

# # # # #     recommendations = []

# # # # #     for idx in indices:
# # # # #         try:
# # # # #             artwork = train_data[int(idx)]  # Convert to int
# # # # #         except IndexError:
# # # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # # #             continue
# # # # #         except TypeError as te:
# # # # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # # #             continue

# # # # #         curr_metadata = {
# # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # #             "description": artwork.get('description', 'No Description Available')
# # # # #         }

# # # # #         image_data_or_url = artwork.get('image', None)

# # # # #         if isinstance(image_data_or_url, str):
# # # # #             try:
# # # # #                 response = requests.get(image_data_or_url)
# # # # #                 if response.status_code == 200:
# # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # #                 else:
# # # # #                     artwork_image = None
# # # # #             except Exception as e:
# # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # #                 artwork_image = None
# # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # #             artwork_image = image_data_or_url
# # # # #         else:
# # # # #             artwork_image = None

# # # # #         if artwork_image:
# # # # #             try:
# # # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # # #             except Exception as e:
# # # # #                 print(f"Error encoding image to base64: {e}")
# # # # #                 img_base64 = None
# # # # #         else:
# # # # #             img_base64 = None

# # # # #         recommendations.append({
# # # # #             'index': int(idx),  # Convert to int
# # # # #             'artist': curr_metadata['artist'],
# # # # #             'style': curr_metadata['style'],
# # # # #             'genre': curr_metadata['genre'],
# # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # #             'image': img_base64
# # # # #         })

# # # # #     return jsonify({'images': recommendations}), 200

# # # # # @app.route('/like-image', methods=['POST'])
# # # # # @token_required
# # # # # def like_image(current_user_email):
# # # # #     """
# # # # #     Records a user's like for an image and updates embeddings.
# # # # #     """
# # # # #     data = request.get_json()
# # # # #     image_index = data.get('image_index')

# # # # #     if image_index is None:
# # # # #         return jsonify({'message': 'Image index is required.'}), 400

# # # # #     # Ensure image_index is int
# # # # #     try:
# # # # #         image_index = int(image_index)
# # # # #     except ValueError:
# # # # #         return jsonify({'message': 'Image index must be an integer.'}), 400

# # # # #     # Record the like
# # # # #     try:
# # # # #         user_likes = load_user_likes()
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # #     # Ensure image_index is within range
# # # # #     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
# # # # #         return jsonify({'message': 'Invalid image index.'}), 400

# # # # #     new_like = pd.DataFrame({
# # # # #         'UserEmail': [current_user_email],
# # # # #         'ImageIndex': [image_index],
# # # # #         'LikedAt': [datetime.datetime.utcnow()]
# # # # #     })

# # # # #     try:
# # # # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # # # #         save_user_likes(user_likes)
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # # # #     # Update user embedding after k likes
# # # # #     k = 5  # Define k as needed
# # # # #     user_total_likes = user_likes[user_likes['UserEmail'] == current_user_email].shape[0]

# # # # #     if user_total_likes % k == 0:
# # # # #         try:
# # # # #             user_embeddings = load_user_embeddings()
# # # # #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # #             if user_embedding_row.empty:
# # # # #                 # Initialize embedding with zeros if not found
# # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # #                 zero_embedding = np.zeros(embedding_dim)
# # # # #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # #                 new_user_embedding = pd.DataFrame({
# # # # #                     'UserEmail': [current_user_email],
# # # # #                     'Embedding': [zero_embedding_encoded],
# # # # #                     'LastRecommendedIndex': [0],
# # # # #                     'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # #                 })
# # # # #                 user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # # #                 save_user_embeddings(user_embeddings)
# # # # #                 user_embedding = zero_embedding
# # # # #                 print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # # # #             else:
# # # # #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # # # #             # Fetch user's liked image embeddings
# # # # #             liked_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()
# # # # #             if all_embeddings is not None and len(liked_indices) > 0:
# # # # #                 # Ensure indices are within range
# # # # #                 liked_indices = [i for i in liked_indices if 0 <= i < all_embeddings.shape[0]]
# # # # #                 if len(liked_indices) > 0:
# # # # #                     liked_embeddings = all_embeddings[liked_indices]
# # # # #                     # Compute the average of liked embeddings
# # # # #                     average_liked_embedding = np.mean(liked_embeddings, axis=0)
# # # # #                     if np.linalg.norm(average_liked_embedding) != 0:
# # # # #                         average_liked_embedding = average_liked_embedding / np.linalg.norm(average_liked_embedding)
# # # # #                 else:
# # # # #                     # If no valid liked_indices, use zero embedding
# # # # #                     embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # #                     average_liked_embedding = np.zeros(embedding_dim)
# # # # #             else:
# # # # #                 # If no embeddings available, use zero embedding
# # # # #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # #                 average_liked_embedding = np.zeros(embedding_dim)

# # # # #             # Combine with previous embedding
# # # # #             combined_embedding = combine_embeddings_for_recommendation(
# # # # #                 average_liked_embedding, user_embedding, weight=0.7
# # # # #             )
# # # # #             norm = np.linalg.norm(combined_embedding)
# # # # #             if norm != 0:
# # # # #                 combined_embedding = combined_embedding / norm
# # # # #             else:
# # # # #                 combined_embedding = combined_embedding

# # # # #             # Update the embedding
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# # # # #             # Reset LastRecommendedIndex since embedding has been updated
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# # # # #             # Update LastEmbeddingUpdate timestamp
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# # # # #             save_user_embeddings(user_embeddings)
# # # # #             print(f"Updated embedding for user {current_user_email} after {k} likes.")
# # # # #         except Exception as e:
# # # # #             return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # # # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # # # @app.route('/recommend-images', methods=['GET'])
# # # # # @token_required
# # # # # def recommend_images(current_user_email):
# # # # #     """
# # # # #     Provides personalized recommendations based on user embeddings.
# # # # #     """
# # # # #     try:
# # # # #         user_embeddings = load_user_embeddings()
# # # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # # #         if user_embedding_row.empty:
# # # # #             # Initialize embedding with zeros if not found
# # # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # # #             zero_embedding = np.zeros(embedding_dim)
# # # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # # #             new_embedding = pd.DataFrame({
# # # # #                 'UserEmail': [current_user_email],
# # # # #                 'Embedding': [zero_embedding_encoded],
# # # # #                 'LastRecommendedIndex': [0],
# # # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # # #             })
# # # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # # #             save_user_embeddings(user_embeddings)
# # # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # # # #         else:
# # # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # # #     # Check if user has liked any images
# # # # #     try:
# # # # #         user_likes = load_user_likes()
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # # #     if not user_liked_images:
# # # # #         # User hasn't liked any images yet, return random 40 images
# # # # #         if train_data is not None:
# # # # #             num_images = len(train_data)
# # # # #             sample_size = 40 if num_images >= 40 else num_images
# # # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # # #         else:
# # # # #             return jsonify({'message': 'No images available.'}), 500
# # # # #     else:
# # # # #         if all_embeddings is None:
# # # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # # #         # Ensure user_embedding has the correct dimension
# # # # #         embedding_dim = all_embeddings.shape[1]
# # # # #         if user_embedding.shape[1] != embedding_dim:
# # # # #             if user_embedding.shape[1] > embedding_dim:
# # # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # # #             else:
# # # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # # #             # Update the embedding in the dataframe
# # # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # # # #             save_user_embeddings(user_embeddings)

# # # # #         # Compute similarities
# # # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # # #         top_indices = similarities.argsort()[0][::-1]

# # # # #         # Exclude already liked images
# # # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # # #         # Fetch LastRecommendedIndex
# # # # #         try:
# # # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # # #         except:
# # # # #             last_recommended_index = 0

# # # # #         # Define batch size
# # # # #         batch_size = 10

# # # # #         # Select the next batch
# # # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # # #         # Update LastRecommendedIndex
# # # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # # #         save_user_embeddings(user_embeddings)

# # # # #     recommendations = []

# # # # #     for idx in indices:
# # # # #         try:
# # # # #             artwork = train_data[int(idx)]  # Convert to int
# # # # #         except IndexError:
# # # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # # #             continue
# # # # #         except TypeError as te:
# # # # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # # #             continue

# # # # #         curr_metadata = {
# # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # #             "description": artwork.get('description', 'No Description Available')
# # # # #         }

# # # # #         image_data_or_url = artwork.get('image', None)

# # # # #         if isinstance(image_data_or_url, str):
# # # # #             try:
# # # # #                 response = requests.get(image_data_or_url)
# # # # #                 if response.status_code == 200:
# # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # #                 else:
# # # # #                     artwork_image = None
# # # # #             except Exception as e:
# # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # #                 artwork_image = None
# # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # #             artwork_image = image_data_or_url
# # # # #         else:
# # # # #             artwork_image = None

# # # # #         if artwork_image:
# # # # #             try:
# # # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # # #             except Exception as e:
# # # # #                 print(f"Error encoding image to base64: {e}")
# # # # #                 img_base64 = None
# # # # #         else:
# # # # #             img_base64 = None

# # # # #         recommendations.append({
# # # # #             'index': int(idx),  # Convert to int
# # # # #             'artist': curr_metadata['artist'],
# # # # #             'style': curr_metadata['style'],
# # # # #             'genre': curr_metadata['genre'],
# # # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # # #             'image': img_base64
# # # # #         })

# # # # #     return jsonify({'images': recommendations}), 200

# # # # # @app.route('/chat', methods=['POST'])
# # # # # @token_required
# # # # # def chat(current_user_email):
# # # # #     """
# # # # #     Handle chat requests with text and optional image.
# # # # #     Processes the inputs and returns a response.
# # # # #     """
# # # # #     text = request.form.get('text', '').strip()
# # # # #     image_file = request.files.get('image', None)

# # # # #     image_data = None
# # # # #     if image_file:
# # # # #         try:
# # # # #             image_bytes = image_file.read()
# # # # #             image = Image.open(io.BytesIO(image_bytes))
# # # # #             image = image.convert('RGB')
# # # # #             image_data = image
# # # # #         except Exception as e:
# # # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # # #     try:
# # # # #         result = predict(text, image_data)
# # # # #         return jsonify(result), 200
# # # # #     except Exception as e:
# # # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # # def predict(text, image_data=None):
# # # # #     """
# # # # #     Process the input text and image, generate recommendations,
# # # # #     and return them with explanations and metadata.
# # # # #     """
# # # # #     if not all([
# # # # #         all_embeddings is not None, 
# # # # #         train_data is not None, 
# # # # #         clip_model is not None, 
# # # # #         clip_processor is not None, 
# # # # #         blip_model is not None, 
# # # # #         blip_processor is not None
# # # # #     ]):
# # # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # # #     input_image = image_data
# # # # #     user_text = text

# # # # #     if input_image:
# # # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # # #         print(f"Generated image caption: {image_caption}")
# # # # #     else:
# # # # #         image_caption = ""

# # # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # # #     print(f"Context-aware text: {context_aware_text}")

# # # # #     if input_image:
# # # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # # #     else:
# # # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # # #     print("Preprocessed inputs for CLIP.")

# # # # #     with torch.no_grad():
# # # # #         if input_image:
# # # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # # #             image_features_np = image_features.cpu().detach().numpy()
# # # # #         else:
# # # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # # #         text_features_np = text_features.cpu().detach().numpy()
# # # # #     print("Generated and normalized image and text features using CLIP.")

# # # # #     weight_img = 0.1
# # # # #     weight_text = 0.9

# # # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # # #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # # #     if norm != 0:
# # # # #         final_embedding = final_embedding / norm
# # # # #     else:
# # # # #         final_embedding = final_embedding
# # # # #     print("Computed final combined embedding.")

# # # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # # #     embedding_dim = all_embeddings.shape[1]
# # # # #     if final_embedding.shape[1] != embedding_dim:
# # # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # # #         if final_embedding.shape[1] > embedding_dim:
# # # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # # #             print("Trimmed final_embedding.")
# # # # #         else:
# # # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # # #             final_embedding = np.hstack((final_embedding, padding))
# # # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # # #     top_n = 10
# # # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # # #     recommendations = []

# # # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # # #         try:
# # # # #             artwork = train_data[int(i)]
# # # # #         except IndexError:
# # # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # # #             continue

# # # # #         curr_metadata = {
# # # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # # #             "style": artwork.get('style', 'Unknown Style'),
# # # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # # #             "description": artwork.get('description', 'No Description Available')
# # # # #         }

# # # # #         image_data_or_url = artwork.get('image', None)

# # # # #         if isinstance(image_data_or_url, str):
# # # # #             try:
# # # # #                 response = requests.get(image_data_or_url)
# # # # #                 if response.status_code == 200:
# # # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # # #                 else:
# # # # #                     artwork_image = None
# # # # #             except Exception as e:
# # # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # # #                 artwork_image = None
# # # # #         elif isinstance(image_data_or_url, Image.Image):
# # # # #             artwork_image = image_data_or_url
# # # # #         else:
# # # # #             artwork_image = None

# # # # #         if artwork_image:
# # # # #             try:
# # # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # # #             except Exception as e:
# # # # #                 print(f"Error encoding image to base64: {e}")
# # # # #                 img_base64 = None
# # # # #         else:
# # # # #             img_base64 = None

# # # # #         recommendations.append({
# # # # #             'rank': rank,
# # # # #             'index': int(i),  # Convert to int
# # # # #             'artist': curr_metadata['artist'],
# # # # #             'style': curr_metadata['style'],
# # # # #             'genre': curr_metadata['genre'],
# # # # #             # 'description': curr_metadata['description'],
# # # # #             'image': img_base64
# # # # #         })

# # # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # # #     return {
# # # # #         'response': response_text,
# # # # #         'recommendations': recommendations
# # # # #     }

# # # # # if __name__ == '__main__':
# # # # #     app.run(debug=True)


# # # # # backend/app.py

# # # # from flask import Flask, request, jsonify
# # # # import pandas as pd
# # # # import os
# # # # from werkzeug.security import generate_password_hash, check_password_hash
# # # # import jwt
# # # # import datetime
# # # # from functools import wraps
# # # # from flask_cors import CORS
# # # # from dotenv import load_dotenv
# # # # import io
# # # # from PIL import Image
# # # # import numpy as np
# # # # import torch
# # # # from transformers import CLIPProcessor, CLIPModel
# # # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # # from datasets import load_dataset
# # # # from sklearn.metrics.pairwise import cosine_similarity
# # # # import requests
# # # # import base64
# # # # import json
# # # # from filelock import FileLock, Timeout

# # # # DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# # # # # Initialize the Excel file if it doesn't exist
# # # # if not os.path.exists(DATABASE_IMAGE_LIKES):
# # # #     try:
# # # #         df = pd.DataFrame(columns=['ImageIndex', 'Users'])  # 'Users' will store a list of user emails
# # # #         df.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# # # #         print(f"Created {DATABASE_IMAGE_LIKES} with columns: ['ImageIndex', 'Users']")
# # # #     except Exception as e:
# # # #         print(f"Error creating {DATABASE_IMAGE_LIKES}: {e}")

# # # # def display_image(image_data):
# # # #     # Function to display images (not used in backend)
# # # #     pass

# # # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # # #     with torch.no_grad():
# # # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # # #     return caption

# # # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # # #     margin = 0.05
# # # #     if sim_image > sim_text + margin:
# # # #         reason = "the style and composition of the input image."
# # # #     elif sim_text > sim_image + margin:
# # # #         reason = "your textual preferences for nature and the specified colors."
# # # #     else:
# # # #         reason = "a balanced combination of both your image and textual preferences."

# # # #     explanation = (
# # # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # # #         f"is recommended {reason} "
# # # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # # #     )
# # # #     return explanation

# # # # def encode_image_to_base64(image):
# # # #     buffered = io.BytesIO()
# # # #     image.save(buffered, format="JPEG")
# # # #     img_bytes = buffered.getvalue()
# # # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # # #     return img_base64

# # # # def decode_embedding(embedding_str):
# # # #     return np.array(json.loads(embedding_str))

# # # # def encode_embedding(embedding_array):
# # # #     return json.dumps(embedding_array.tolist())

# # # # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# # # #     """
# # # #     Combines the current embedding with the previous one using a weighted average.
# # # #     """
# # # #     if previous_embedding is None:
# # # #         return current_embedding
# # # #     return weight * current_embedding + (1 - weight) * previous_embedding

# # # # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# # # #     """
# # # #     Recommends the top-k similar artworks based on cosine similarity.
# # # #     """
# # # #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# # # #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# # # #     return top_k_indices

# # # # # Determine the device to use for computation
# # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # print(f"Using device: {device}")

# # # # # Load combined embeddings
# # # # try:
# # # #     all_embeddings = np.load('combined_embeddings.npy')
# # # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # # except FileNotFoundError:
# # # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # # #     all_embeddings = None

# # # # if all_embeddings is not None:
# # # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # # #     print("Normalized all_embeddings for cosine similarity.")
# # # # else:
# # # #     print("Skipping normalization due to missing embeddings.")

# # # # # Load the dataset (e.g., WikiArt for training data)
# # # # try:
# # # #     ds = load_dataset("Artificio/WikiArt")
# # # #     train_data = ds['train']
# # # #     print("Loaded WikiArt dataset successfully.")
# # # # except Exception as e:
# # # #     print(f"Error loading dataset: {e}")
# # # #     train_data = None

# # # # # Load CLIP model and processor
# # # # try:
# # # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # # #     clip_model.to(device)
# # # #     print("Loaded CLIP model and processor successfully.")
# # # # except Exception as e:
# # # #     print(f"Error loading CLIP model: {e}")
# # # #     clip_model = None
# # # #     clip_processor = None

# # # # # Load BLIP model and processor for image captioning
# # # # try:
# # # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # # #     blip_model.to(device)
# # # #     print("Loaded BLIP model and processor successfully.")
# # # # except Exception as e:
# # # #     print(f"Error loading BLIP model: {e}")
# # # #     blip_model = None
# # # #     blip_processor = None

# # # # # Load environment variables from .env file
# # # # load_dotenv()

# # # # app = Flask(__name__)
# # # # CORS(app)

# # # # # Retrieve the secret key from environment variables
# # # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # # Ensure that the secret key is set
# # # # if not app.config['SECRET_KEY']:
# # # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # # Define Excel database files
# # # # DATABASE_USERS = 'users.xlsx'
# # # # DATABASE_LIKES = 'user_likes.xlsx'
# # # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'

# # # # # Initialize Excel files if they don't exist
# # # # for db_file, columns in [
# # # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # # ]:
# # # #     if not os.path.exists(db_file):
# # # #         try:
# # # #             df = pd.DataFrame(columns=columns)
# # # #             df.to_excel(db_file, index=False, engine='openpyxl')
# # # #             print(f"Created {db_file} with columns: {columns}")
# # # #         except Exception as e:
# # # #             print(f"Error creating {db_file}: {e}")

# # # # def load_users():
# # # #     lock_path = DATABASE_USERS + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             return pd.read_excel(DATABASE_USERS, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for loading users.")
# # # #     except Exception as e:
# # # #         # Handle decompression errors or other read errors
# # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # #             print(f"Corrupted {DATABASE_USERS}. Recreating it.")
# # # #             df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # # #             try:
# # # #                 df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # #                 print(f"Recreated {DATABASE_USERS}.")
# # # #             except Exception as ex:
# # # #                 print(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # # #                 raise Exception(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # # #             return df
# # # #         else:
# # # #             raise e

# # # # def save_users(df):
# # # #     lock_path = DATABASE_USERS + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for saving users.")
# # # #     except Exception as e:
# # # #         raise e

# # # # def load_image_likes():
# # # #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
# # # #             # Ensure 'Users' column is a list
# # # #             df['Users'] = df['Users'].apply(lambda x: x if isinstance(x, list) else [])
# # # #             return df
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for loading image likes.")
# # # #     except Exception as e:
# # # #         # Handle decompression errors or other read errors
# # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # #             print(f"Corrupted {DATABASE_IMAGE_LIKES}. Recreating it.")
# # # #             df = pd.DataFrame(columns=['ImageIndex', 'Users'])
# # # #             try:
# # # #                 df.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# # # #                 print(f"Recreated {DATABASE_IMAGE_LIKES}.")
# # # #             except Exception as ex:
# # # #                 print(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# # # #                 raise Exception(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# # # #             return df
# # # #         else:
# # # #             raise e

# # # # def save_image_likes(df):
# # # #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             df.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for saving image likes.")
# # # #     except Exception as e:
# # # #         raise e


# # # # def load_user_likes():
# # # #     lock_path = DATABASE_LIKES + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             return pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for loading user likes.")
# # # #     except Exception as e:
# # # #         # Handle decompression errors
# # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # #             print(f"Corrupted {DATABASE_LIKES}. Recreating it.")
# # # #             df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# # # #             try:
# # # #                 df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # #                 print(f"Recreated {DATABASE_LIKES}.")
# # # #             except Exception as ex:
# # # #                 print(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # # #                 raise Exception(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # # #             return df
# # # #         else:
# # # #             raise e

# # # # def save_user_likes(df):
# # # #     lock_path = DATABASE_LIKES + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for saving user likes.")
# # # #     except Exception as e:
# # # #         raise e

# # # # def load_user_embeddings():
# # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             return pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for loading user embeddings.")
# # # #     except Exception as e:
# # # #         # Handle decompression errors
# # # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # # #             print(f"Corrupted {DATABASE_EMBEDDINGS}. Recreating it.")
# # # #             df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # # #             try:
# # # #                 df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # #                 print(f"Recreated {DATABASE_EMBEDDINGS}.")
# # # #             except Exception as ex:
# # # #                 print(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # # #                 raise Exception(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # # #             return df
# # # #         else:
# # # #             raise e

# # # # def save_user_embeddings(df):
# # # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # # #     lock = FileLock(lock_path)
# # # #     try:
# # # #         with lock.acquire(timeout=10):
# # # #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # # #     except Timeout:
# # # #         raise Exception("Could not acquire lock for saving user embeddings.")
# # # #     except Exception as e:
# # # #         raise e

# # # # def token_required(f):
# # # #     @wraps(f)
# # # #     def decorated(*args, **kwargs):
# # # #         token = None

# # # #         if 'Authorization' in request.headers:
# # # #             auth_header = request.headers['Authorization']
# # # #             try:
# # # #                 token = auth_header.split(" ")[1]
# # # #             except IndexError:
# # # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # # #         if not token:
# # # #             return jsonify({'message': 'Token is missing!'}), 401

# # # #         try:
# # # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # # #             current_user_email = data['email']
# # # #         except jwt.ExpiredSignatureError:
# # # #             return jsonify({'message': 'Token has expired!'}), 401
# # # #         except jwt.InvalidTokenError:
# # # #             return jsonify({'message': 'Invalid token!'}), 401

# # # #         try:
# # # #             users = load_users()
# # # #         except Exception as e:
# # # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # #         user = users[users['Email'] == current_user_email]
# # # #         if user.empty:
# # # #             return jsonify({'message': 'User not found!'}), 401

# # # #         return f(current_user_email, *args, **kwargs)

# # # #     return decorated

# # # # @app.route('/signup', methods=['POST'])
# # # # def signup():
# # # #     data = request.get_json()
# # # #     full_name = data.get('full_name')
# # # #     email = data.get('email')
# # # #     password = data.get('password')

# # # #     if not all([full_name, email, password]):
# # # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # # #     try:
# # # #         users = load_users()
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # #     if email in users['Email'].values:
# # # #         return jsonify({'message': 'Email already exists.'}), 400

# # # #     hashed_password = generate_password_hash(password)

# # # #     new_user = pd.DataFrame({
# # # #         'FullName': [full_name],
# # # #         'Email': [email],
# # # #         'Password': [hashed_password]
# # # #     })

# # # #     try:
# # # #         users = pd.concat([users, new_user], ignore_index=True)
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # # #     try:
# # # #         save_users(users)
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # # #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# # # #     try:
# # # #         user_embeddings = load_user_embeddings()
# # # #         if email not in user_embeddings['UserEmail'].values:
# # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # #             zero_embedding = np.zeros(embedding_dim)
# # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # #             new_embedding = pd.DataFrame({
# # # #                 'UserEmail': [email],
# # # #                 'Embedding': [zero_embedding_encoded],
# # # #                 'LastRecommendedIndex': [0],
# # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # #             })
# # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # #             save_user_embeddings(user_embeddings)
# # # #             print(f"Initialized zero embedding for user {email}.")
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # # @app.route('/trending', methods=['GET'])
# # # # @token_required
# # # # def trending(current_user_email):
# # # #     """
# # # #     Retrieves the top 40 trending images based on the number of likes.
# # # #     Returns the images along with their like counts.
# # # #     """
# # # #     try:
# # # #         image_likes = load_image_likes()
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

# # # #     # Calculate like counts for each image
# # # #     image_likes['LikeCount'] = image_likes['Users'].apply(len)

# # # #     # Sort images by LikeCount descendingly
# # # #     top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

# # # #     recommendations = []

# # # #     for _, row in top_images.iterrows():
# # # #         idx = row['ImageIndex']
# # # #         like_count = row['LikeCount']

# # # #         try:
# # # #             artwork = train_data[int(idx)]
# # # #         except IndexError:
# # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # #             continue
# # # #         except TypeError as te:
# # # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # #             continue

# # # #         curr_metadata = {
# # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # #             "style": artwork.get('style', 'Unknown Style'),
# # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # #             "description": artwork.get('description', 'No Description Available')
# # # #         }

# # # #         image_data_or_url = artwork.get('image', None)

# # # #         if isinstance(image_data_or_url, str):
# # # #             try:
# # # #                 response = requests.get(image_data_or_url)
# # # #                 if response.status_code == 200:
# # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # #                 else:
# # # #                     artwork_image = None
# # # #             except Exception as e:
# # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # #                 artwork_image = None
# # # #         elif isinstance(image_data_or_url, Image.Image):
# # # #             artwork_image = image_data_or_url
# # # #         else:
# # # #             artwork_image = None

# # # #         if artwork_image:
# # # #             try:
# # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # #             except Exception as e:
# # # #                 print(f"Error encoding image to base64: {e}")
# # # #                 img_base64 = None
# # # #         else:
# # # #             img_base64 = None

# # # #         recommendations.append({
# # # #             'index': int(idx),  # Convert to int
# # # #             'artist': curr_metadata['artist'],
# # # #             'style': curr_metadata['style'],
# # # #             'genre': curr_metadata['genre'],
# # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # #             'image': img_base64,
# # # #             'like_count': like_count
# # # #         })

# # # #     return jsonify({'trending_images': recommendations}), 200


# # # # @app.route('/login', methods=['POST'])
# # # # def login():
# # # #     data = request.get_json()
# # # #     email = data.get('email')
# # # #     password = data.get('password')

# # # #     if not all([email, password]):
# # # #         return jsonify({'message': 'Email and password are required.'}), 400

# # # #     try:
# # # #         users = load_users()
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # # #     user = users[users['Email'] == email]

# # # #     if user.empty:
# # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # #     stored_password = user.iloc[0]['Password']
# # # #     full_name = user.iloc[0]['FullName']

# # # #     if not check_password_hash(stored_password, password):
# # # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # # #     try:
# # # #         token = jwt.encode({
# # # #             'email': email,
# # # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # # #     # Ensure user has an embedding; initialize if not
# # # #     try:
# # # #         user_embeddings = load_user_embeddings()
# # # #         if email not in user_embeddings['UserEmail'].values:
# # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # # #             zero_embedding = np.zeros(embedding_dim)
# # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # #             new_embedding = pd.DataFrame({
# # # #                 'UserEmail': [email],
# # # #                 'Embedding': [zero_embedding_encoded],
# # # #                 'LastRecommendedIndex': [0],
# # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # #             })
# # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # #             save_user_embeddings(user_embeddings)
# # # #             print(f"Initialized zero embedding for user {email} on login.")
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # # @app.route('/protected', methods=['GET'])
# # # # @token_required
# # # # def protected_route(current_user_email):
# # # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # # --- Removed the /get-images Endpoint ---
# # # # # Since we are now using only /recommend-images, we remove /get-images.
# # # # # Uncomment the following lines if you want to retain /get-images for any reason.

# # # # # @app.route('/get-images', methods=['GET'])
# # # # # @token_required
# # # # # def get_images(current_user_email):
# # # # #     """
# # # # #     Fetch a batch of images for the user.
# # # # #     If the user has not liked any images, return random indices.
# # # # #     Otherwise, fetch based on recommendations.
# # # # #     """
# # # # #     # ... existing code ...
# # # # #     pass  # This endpoint has been removed as per requirements.

# # # # # --- Modified /like-image Endpoint ---
# # # # @app.route('/like-image', methods=['POST'])
# # # # @token_required
# # # # def like_image(current_user_email):
# # # #     """
# # # #     Records a user's like for an image and updates embeddings immediately.
# # # #     """
# # # #     data = request.get_json()
# # # #     image_index = data.get('image_index')

# # # #     if image_index is None:
# # # #         return jsonify({'message': 'Image index is required.'}), 400

# # # #     # Ensure image_index is int
# # # #     try:
# # # #         image_index = int(image_index)
# # # #     except ValueError:
# # # #         return jsonify({'message': 'Image index must be an integer.'}), 400

# # # #     # Record the like
# # # #     try:
# # # #         user_likes = load_user_likes()
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # #     # Ensure image_index is within range
# # # #     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
# # # #         return jsonify({'message': 'Invalid image index.'}), 400

# # # #     new_like = pd.DataFrame({
# # # #         'UserEmail': [current_user_email],
# # # #         'ImageIndex': [image_index],
# # # #         'LikedAt': [datetime.datetime.utcnow()]
# # # #     })

# # # #     try:
# # # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # # #         save_user_likes(user_likes)
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # # #     # --- Update user embedding immediately after each like ---
# # # #     try:
# # # #         user_embeddings = load_user_embeddings()
# # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # #         if user_embedding_row.empty:
# # # #             # Initialize embedding with zeros if not found
# # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # #             zero_embedding = np.zeros(embedding_dim)
# # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # #             new_user_embedding = pd.DataFrame({
# # # #                 'UserEmail': [current_user_email],
# # # #                 'Embedding': [zero_embedding_encoded],
# # # #                 'LastRecommendedIndex': [0],
# # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # #             })
# # # #             user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # # #             save_user_embeddings(user_embeddings)
# # # #             user_embedding = zero_embedding
# # # #             print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # # #         else:
# # # #             # Decode the existing embedding
# # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # # #         # Fetch the liked image embedding
# # # #         if all_embeddings is not None:
# # # #             liked_embedding = all_embeddings[image_index]
# # # #             if np.linalg.norm(liked_embedding) != 0:
# # # #                 liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
# # # #             else:
# # # #                 liked_embedding = liked_embedding
# # # #         else:
# # # #             # If embeddings are not available, use zero embedding
# # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # #             liked_embedding = np.zeros(embedding_dim)

# # # #         # Combine with previous embedding using the specified weight
# # # #         weight = 0.7  # Define the weight as needed
# # # #         combined_embedding = combine_embeddings_for_recommendation(
# # # #             current_embedding=liked_embedding,
# # # #             previous_embedding=user_embedding,
# # # #             weight=weight
# # # #         )
# # # #         norm = np.linalg.norm(combined_embedding)
# # # #         if norm != 0:
# # # #             combined_embedding = combined_embedding / norm
# # # #         else:
# # # #             combined_embedding = combined_embedding

# # # #         # Update the embedding in the dataframe
# # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# # # #         # Reset LastRecommendedIndex since embedding has been updated
# # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# # # #         # Update LastEmbeddingUpdate timestamp
# # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# # # #         # Save the updated embeddings
# # # #         save_user_embeddings(user_embeddings)
# # # #         print(f"Updated embedding for user {current_user_email} after like.")

# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # # @app.route('/recommend-images', methods=['GET'])
# # # # @token_required
# # # # def recommend_images(current_user_email):
# # # #     """
# # # #     Provides personalized recommendations based on user embeddings.
# # # #     """
# # # #     try:
# # # #         user_embeddings = load_user_embeddings()
# # # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # # #         if user_embedding_row.empty:
# # # #             # Initialize embedding with zeros if not found
# # # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # # #             zero_embedding = np.zeros(embedding_dim)
# # # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # # #             new_embedding = pd.DataFrame({
# # # #                 'UserEmail': [current_user_email],
# # # #                 'Embedding': [zero_embedding_encoded],
# # # #                 'LastRecommendedIndex': [0],
# # # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # # #             })
# # # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # # #             save_user_embeddings(user_embeddings)
# # # #             user_embedding = zero_embedding.reshape(1, -1)
# # # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # # #         else:
# # # #             # Decode the existing embedding
# # # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # # #     # Check if user has liked any images
# # # #     try:
# # # #         user_likes = load_user_likes()
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # #     if not user_liked_images:
# # # #         # User hasn't liked any images yet, return random 40 images
# # # #         if train_data is not None:
# # # #             num_images = len(train_data)
# # # #             sample_size = 10 if num_images >= 10 else num_images
# # # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # # #         else:
# # # #             return jsonify({'message': 'No images available.'}), 500
# # # #     else:
# # # #         if all_embeddings is None:
# # # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # # #         # Ensure user_embedding has the correct dimension
# # # #         embedding_dim = all_embeddings.shape[1]
# # # #         if user_embedding.shape[1] != embedding_dim:
# # # #             if user_embedding.shape[1] > embedding_dim:
# # # #                 user_embedding = user_embedding[:, :embedding_dim]
# # # #                 print("Trimmed user_embedding to match embedding_dim.")
# # # #             else:
# # # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # # #                 user_embedding = np.hstack((user_embedding, padding))
# # # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # # #             # Update the embedding in the dataframe
# # # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # # #             save_user_embeddings(user_embeddings)

# # # #         # Compute similarities
# # # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # # #         top_indices = similarities.argsort()[0][::-1]

# # # #         # Exclude already liked images
# # # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # # #         # Fetch LastRecommendedIndex
# # # #         try:
# # # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # # #         except:
# # # #             last_recommended_index = 0

# # # #         # Define batch size
# # # #         batch_size = 10

# # # #         # Select the next batch
# # # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # # #         # Update LastRecommendedIndex
# # # #         new_last_recommended_index = last_recommended_index + batch_size
# # # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # # #         save_user_embeddings(user_embeddings)

# # # #     recommendations = []

# # # #     for idx in indices:
# # # #         try:
# # # #             artwork = train_data[int(idx)]  # Convert to int
# # # #         except IndexError:
# # # #             print(f"Index {idx} is out of bounds for the dataset.")
# # # #             continue
# # # #         except TypeError as te:
# # # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # #             continue

# # # #         curr_metadata = {
# # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # #             "style": artwork.get('style', 'Unknown Style'),
# # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # #             "description": artwork.get('description', 'No Description Available')
# # # #         }

# # # #         image_data_or_url = artwork.get('image', None)

# # # #         if isinstance(image_data_or_url, str):
# # # #             try:
# # # #                 response = requests.get(image_data_or_url)
# # # #                 if response.status_code == 200:
# # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # #                 else:
# # # #                     artwork_image = None
# # # #             except Exception as e:
# # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # #                 artwork_image = None
# # # #         elif isinstance(image_data_or_url, Image.Image):
# # # #             artwork_image = image_data_or_url
# # # #         else:
# # # #             artwork_image = None

# # # #         if artwork_image:
# # # #             try:
# # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # #             except Exception as e:
# # # #                 print(f"Error encoding image to base64: {e}")
# # # #                 img_base64 = None
# # # #         else:
# # # #             img_base64 = None

# # # #         recommendations.append({
# # # #             'index': int(idx),  # Convert to int
# # # #             'artist': curr_metadata['artist'],
# # # #             'style': curr_metadata['style'],
# # # #             'genre': curr_metadata['genre'],
# # # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # #             'image': img_base64
# # # #         })

# # # #     return jsonify({'images': recommendations}), 200

# # # # @app.route('/chat', methods=['POST'])
# # # # @token_required
# # # # def chat(current_user_email):
# # # #     """
# # # #     Handle chat requests with text and optional image.
# # # #     Processes the inputs and returns a response.
# # # #     """
# # # #     text = request.form.get('text', '').strip()
# # # #     image_file = request.files.get('image', None)

# # # #     image_data = None
# # # #     if image_file:
# # # #         try:
# # # #             image_bytes = image_file.read()
# # # #             image = Image.open(io.BytesIO(image_bytes))
# # # #             image = image.convert('RGB')
# # # #             image_data = image
# # # #         except Exception as e:
# # # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # # #     try:
# # # #         result = predict(text, image_data)
# # # #         return jsonify(result), 200
# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # # def predict(text, image_data=None):
# # # #     """
# # # #     Process the input text and image, generate recommendations,
# # # #     and return them with explanations and metadata.
# # # #     """
# # # #     if not all([
# # # #         all_embeddings is not None, 
# # # #         train_data is not None, 
# # # #         clip_model is not None, 
# # # #         clip_processor is not None, 
# # # #         blip_model is not None, 
# # # #         blip_processor is not None
# # # #     ]):
# # # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # # #     input_image = image_data
# # # #     user_text = text

# # # #     if input_image:
# # # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # # #         print(f"Generated image caption: {image_caption}")
# # # #     else:
# # # #         image_caption = ""

# # # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # # #     print(f"Context-aware text: {context_aware_text}")

# # # #     if input_image:
# # # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # # #     else:
# # # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # # #     print("Preprocessed inputs for CLIP.")

# # # #     with torch.no_grad():
# # # #         if input_image:
# # # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # # #             image_features_np = image_features.cpu().detach().numpy()
# # # #         else:
# # # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # # #         text_features_np = text_features.cpu().detach().numpy()
# # # #     print("Generated and normalized image and text features using CLIP.")

# # # #     weight_img = 0.1
# # # #     weight_text = 0.9

# # # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # # #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # # #     if norm != 0:
# # # #         final_embedding = final_embedding / norm
# # # #     else:
# # # #         final_embedding = final_embedding
# # # #     print("Computed final combined embedding.")

# # # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # # #     embedding_dim = all_embeddings.shape[1]
# # # #     if final_embedding.shape[1] != embedding_dim:
# # # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # # #         if final_embedding.shape[1] > embedding_dim:
# # # #             final_embedding = final_embedding[:, :embedding_dim]
# # # #             print("Trimmed final_embedding.")
# # # #         else:
# # # #             padding_size = embedding_dim - final_embedding.shape[1]
# # # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # # #             final_embedding = np.hstack((final_embedding, padding))
# # # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # # #     top_n = 10
# # # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # # #     recommended_artworks = [int(i) for i in top_n_indices]

# # # #     recommendations = []

# # # #     for rank, i in enumerate(recommended_artworks, start=1):
# # # #         try:
# # # #             artwork = train_data[int(i)]
# # # #         except IndexError:
# # # #             print(f"Index {i} is out of bounds for the dataset.")
# # # #             continue

# # # #         curr_metadata = {
# # # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # # #             "style": artwork.get('style', 'Unknown Style'),
# # # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # # #             "description": artwork.get('description', 'No Description Available')
# # # #         }

# # # #         image_data_or_url = artwork.get('image', None)

# # # #         if isinstance(image_data_or_url, str):
# # # #             try:
# # # #                 response = requests.get(image_data_or_url)
# # # #                 if response.status_code == 200:
# # # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # #                 else:
# # # #                     artwork_image = None
# # # #             except Exception as e:
# # # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # # #                 artwork_image = None
# # # #         elif isinstance(image_data_or_url, Image.Image):
# # # #             artwork_image = image_data_or_url
# # # #         else:
# # # #             artwork_image = None

# # # #         if artwork_image:
# # # #             try:
# # # #                 img_base64 = encode_image_to_base64(artwork_image)
# # # #             except Exception as e:
# # # #                 print(f"Error encoding image to base64: {e}")
# # # #                 img_base64 = None
# # # #         else:
# # # #             img_base64 = None

# # # #         recommendations.append({
# # # #             'rank': rank,
# # # #             'index': int(i),  # Convert to int
# # # #             'artist': curr_metadata['artist'],
# # # #             'style': curr_metadata['style'],
# # # #             'genre': curr_metadata['genre'],
# # # #             # 'description': curr_metadata['description'],  # Optional: Uncomment if needed
# # # #             'image': img_base64
# # # #         })

# # # #     response_text = "Here are the recommended artworks based on your preferences:"

# # # #     return {
# # # #         'response': response_text,
# # # #         'recommendations': recommendations
# # # #     }

# # # # @app.route('/get_all_liked', methods=['GET'])
# # # # @token_required
# # # # def get_all_liked(current_user_email):
# # # #     """
# # # #     Retrieves all liked images for the authenticated user.
# # # #     """
# # # #     try:
# # # #         # Load user likes
# # # #         user_likes = load_user_likes()
# # # #         liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # # #         if not liked_image_indices:
# # # #             return jsonify({'liked_images': []}), 200

# # # #         # Fetch image data from train_data
# # # #         liked_images = []
# # # #         for idx in liked_image_indices:
# # # #             try:
# # # #                 artwork = train_data[int(idx)]
# # # #             except IndexError:
# # # #                 print(f"Index {idx} is out of bounds for the dataset.")
# # # #                 continue
# # # #             except TypeError as te:
# # # #                 print(f"TypeError accessing train_data with idx={idx}: {te}")
# # # #                 continue

# # # #             curr_metadata = {
# # # #                 "artist": artwork.get('artist', 'Unknown Artist'),
# # # #                 "style": artwork.get('style', 'Unknown Style'),
# # # #                 "genre": artwork.get('genre', 'Unknown Genre'),
# # # #                 "description": artwork.get('description', 'No Description Available')
# # # #             }

# # # #             image_data_or_url = artwork.get('image', None)

# # # #             if isinstance(image_data_or_url, str):
# # # #                 try:
# # # #                     response = requests.get(image_data_or_url)
# # # #                     if response.status_code == 200:
# # # #                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # # #                     else:
# # # #                         artwork_image = None
# # # #                 except Exception as e:
# # # #                     print(f"Error fetching image from {image_data_or_url}: {e}")
# # # #                     artwork_image = None
# # # #             elif isinstance(image_data_or_url, Image.Image):
# # # #                 artwork_image = image_data_or_url
# # # #             else:
# # # #                 artwork_image = None

# # # #             if artwork_image:
# # # #                 try:
# # # #                     img_base64 = encode_image_to_base64(artwork_image)
# # # #                 except Exception as e:
# # # #                     print(f"Error encoding image to base64: {e}")
# # # #                     img_base64 = None
# # # #             else:
# # # #                 img_base64 = None

# # # #             liked_images.append({
# # # #                 'index': int(idx),  # Convert to int
# # # #                 'artist': curr_metadata['artist'],
# # # #                 'style': curr_metadata['style'],
# # # #                 'genre': curr_metadata['genre'],
# # # #                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # # #                 'image': img_base64,
# # # #                 'timestamp': datetime.datetime.utcnow().isoformat()  # Optional: Add timestamp if needed
# # # #             })

# # # #         return jsonify({'liked_images': liked_images}), 200

# # # #     except Exception as e:
# # # #         return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500


# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)


# # # # backend/app.py

# # # from flask import Flask, request, jsonify
# # # import pandas as pd
# # # import os
# # # from werkzeug.security import generate_password_hash, check_password_hash
# # # import jwt
# # # import datetime
# # # from functools import wraps
# # # from flask_cors import CORS
# # # from dotenv import load_dotenv
# # # import io
# # # from PIL import Image
# # # import numpy as np
# # # import torch
# # # from transformers import CLIPProcessor, CLIPModel
# # # from transformers import BlipProcessor, BlipForConditionalGeneration
# # # from datasets import load_dataset
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # import requests
# # # import base64
# # # import json
# # # from filelock import FileLock, Timeout

# # # def display_image(image_data):
# # #     # Function to display images (not used in backend)
# # #     pass

# # # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# # #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# # #     with torch.no_grad():
# # #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# # #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# # #     return caption

# # # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# # #     margin = 0.05
# # #     if sim_image > sim_text + margin:
# # #         reason = "the style and composition of the input image."
# # #     elif sim_text > sim_image + margin:
# # #         reason = "your textual preferences for nature and the specified colors."
# # #     else:
# # #         reason = "a balanced combination of both your image and textual preferences."

# # #     explanation = (
# # #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# # #         f"is recommended {reason} "
# # #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# # #     )
# # #     return explanation

# # # def encode_image_to_base64(image):
# # #     buffered = io.BytesIO()
# # #     image.save(buffered, format="JPEG")
# # #     img_bytes = buffered.getvalue()
# # #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# # #     return img_base64

# # # def decode_embedding(embedding_str):
# # #     return np.array(json.loads(embedding_str))

# # # def encode_embedding(embedding_array):
# # #     return json.dumps(embedding_array.tolist())

# # # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# # #     """
# # #     Combines the current embedding with the previous one using a weighted average.
# # #     """
# # #     if previous_embedding is None:
# # #         return current_embedding
# # #     return weight * current_embedding + (1 - weight) * previous_embedding

# # # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# # #     """
# # #     Recommends the top-k similar artworks based on cosine similarity.
# # #     """
# # #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# # #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# # #     return top_k_indices

# # # # Determine the device to use for computation
# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # print(f"Using device: {device}")

# # # # Load combined embeddings
# # # try:
# # #     all_embeddings = np.load('combined_embeddings.npy')
# # #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # # except FileNotFoundError:
# # #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# # #     all_embeddings = None

# # # if all_embeddings is not None:
# # #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# # #     print("Normalized all_embeddings for cosine similarity.")
# # # else:
# # #     print("Skipping normalization due to missing embeddings.")

# # # # Load the dataset (e.g., WikiArt for training data)
# # # try:
# # #     ds = load_dataset("Artificio/WikiArt")
# # #     train_data = ds['train']
# # #     print("Loaded WikiArt dataset successfully.")
# # # except Exception as e:
# # #     print(f"Error loading dataset: {e}")
# # #     train_data = None

# # # # Load CLIP model and processor
# # # try:
# # #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # #     clip_model.to(device)
# # #     print("Loaded CLIP model and processor successfully.")
# # # except Exception as e:
# # #     print(f"Error loading CLIP model: {e}")
# # #     clip_model = None
# # #     clip_processor = None

# # # # Load BLIP model and processor for image captioning
# # # try:
# # #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# # #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# # #     blip_model.to(device)
# # #     print("Loaded BLIP model and processor successfully.")
# # # except Exception as e:
# # #     print(f"Error loading BLIP model: {e}")
# # #     blip_model = None
# # #     blip_processor = None

# # # # Load environment variables from .env file
# # # load_dotenv()

# # # app = Flask(__name__)
# # # CORS(app)

# # # # Retrieve the secret key from environment variables
# # # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # # Ensure that the secret key is set
# # # if not app.config['SECRET_KEY']:
# # #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # # Define Excel database files
# # # DATABASE_USERS = 'users.xlsx'
# # # DATABASE_LIKES = 'user_likes.xlsx'
# # # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'
# # # DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# # # # Initialize Excel files if they don't exist
# # # for db_file, columns in [
# # #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# # #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# # #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate']),
# # #     (DATABASE_IMAGE_LIKES, ['ImageIndex', 'Users'])  # 'Users' will store JSON strings
# # # ]:
# # #     if not os.path.exists(db_file):
# # #         try:
# # #             if db_file == DATABASE_IMAGE_LIKES:
# # #                 df = pd.DataFrame(columns=columns)
# # #             else:
# # #                 df = pd.DataFrame(columns=columns)
# # #             df.to_excel(db_file, index=False, engine='openpyxl')
# # #             print(f"Created {db_file} with columns: {columns}")
# # #         except Exception as e:
# # #             print(f"Error creating {db_file}: {e}")

# # # def load_users():
# # #     lock_path = DATABASE_USERS + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             return pd.read_excel(DATABASE_USERS, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for loading users.")
# # #     except Exception as e:
# # #         # Handle decompression errors or other read errors
# # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # #             print(f"Corrupted {DATABASE_USERS}. Recreating it.")
# # #             df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# # #             try:
# # #                 df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # #                 print(f"Recreated {DATABASE_USERS}.")
# # #             except Exception as ex:
# # #                 print(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # #                 raise Exception(f"Failed to recreate {DATABASE_USERS}: {ex}")
# # #             return df
# # #         else:
# # #             raise e

# # # def save_users(df):
# # #     lock_path = DATABASE_USERS + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for saving users.")
# # #     except Exception as e:
# # #         raise e

# # # def load_user_likes():
# # #     lock_path = DATABASE_LIKES + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             return pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for loading user likes.")
# # #     except Exception as e:
# # #         # Handle decompression errors
# # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # #             print(f"Corrupted {DATABASE_LIKES}. Recreating it.")
# # #             df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# # #             try:
# # #                 df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # #                 print(f"Recreated {DATABASE_LIKES}.")
# # #             except Exception as ex:
# # #                 print(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # #                 raise Exception(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# # #             return df
# # #         else:
# # #             raise e

# # # def save_user_likes(df):
# # #     lock_path = DATABASE_LIKES + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for saving user likes.")
# # #     except Exception as e:
# # #         raise e

# # # def load_user_embeddings():
# # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             return pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for loading user embeddings.")
# # #     except Exception as e:
# # #         # Handle decompression errors
# # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # #             print(f"Corrupted {DATABASE_EMBEDDINGS}. Recreating it.")
# # #             df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# # #             try:
# # #                 df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # #                 print(f"Recreated {DATABASE_EMBEDDINGS}.")
# # #             except Exception as ex:
# # #                 print(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # #                 raise Exception(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# # #             return df
# # #         else:
# # #             raise e

# # # def save_user_embeddings(df):
# # #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for saving user embeddings.")
# # #     except Exception as e:
# # #         raise e

# # # def load_image_likes():
# # #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         with lock.acquire(timeout=10):
# # #             df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
# # #             # Ensure 'Users' column is parsed from JSON strings to lists
# # #             df['Users'] = df['Users'].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
# # #             return df
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for loading image likes.")
# # #     except Exception as e:
# # #         # Handle decompression errors or other read errors
# # #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# # #             print(f"Corrupted {DATABASE_IMAGE_LIKES}. Recreating it.")
# # #             df = pd.DataFrame(columns=['ImageIndex', 'Users'])
# # #             try:
# # #                 df.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# # #                 print(f"Recreated {DATABASE_IMAGE_LIKES}.")
# # #             except Exception as ex:
# # #                 print(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# # #                 raise Exception(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# # #             return df
# # #         else:
# # #             raise e

# # # def save_image_likes(df):
# # #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# # #     lock = FileLock(lock_path)
# # #     try:
# # #         # Convert 'Users' lists to JSON strings before saving
# # #         df_copy = df.copy()
# # #         df_copy['Users'] = df_copy['Users'].apply(lambda x: json.dumps(x))
# # #         with lock.acquire(timeout=10):
# # #             df_copy.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# # #     except Timeout:
# # #         raise Exception("Could not acquire lock for saving image likes.")
# # #     except Exception as e:
# # #         raise e

# # # def token_required(f):
# # #     @wraps(f)
# # #     def decorated(*args, **kwargs):
# # #         token = None

# # #         if 'Authorization' in request.headers:
# # #             auth_header = request.headers['Authorization']
# # #             try:
# # #                 token = auth_header.split(" ")[1]
# # #             except IndexError:
# # #                 return jsonify({'message': 'Token format invalid!'}), 401

# # #         if not token:
# # #             return jsonify({'message': 'Token is missing!'}), 401

# # #         try:
# # #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# # #             current_user_email = data['email']
# # #         except jwt.ExpiredSignatureError:
# # #             return jsonify({'message': 'Token has expired!'}), 401
# # #         except jwt.InvalidTokenError:
# # #             return jsonify({'message': 'Invalid token!'}), 401

# # #         try:
# # #             users = load_users()
# # #         except Exception as e:
# # #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # #         user = users[users['Email'] == current_user_email]
# # #         if user.empty:
# # #             return jsonify({'message': 'User not found!'}), 401

# # #         return f(current_user_email, *args, **kwargs)

# # #     return decorated

# # # @app.route('/signup', methods=['POST'])
# # # def signup():
# # #     data = request.get_json()
# # #     full_name = data.get('full_name')
# # #     email = data.get('email')
# # #     password = data.get('password')

# # #     if not all([full_name, email, password]):
# # #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# # #     try:
# # #         users = load_users()
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # #     if email in users['Email'].values:
# # #         return jsonify({'message': 'Email already exists.'}), 400

# # #     hashed_password = generate_password_hash(password)

# # #     new_user = pd.DataFrame({
# # #         'FullName': [full_name],
# # #         'Email': [email],
# # #         'Password': [hashed_password]
# # #     })

# # #     try:
# # #         users = pd.concat([users, new_user], ignore_index=True)
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# # #     try:
# # #         save_users(users)
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# # #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# # #     try:
# # #         user_embeddings = load_user_embeddings()
# # #         if email not in user_embeddings['UserEmail'].values:
# # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # #             zero_embedding = np.zeros(embedding_dim)
# # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # #             new_embedding = pd.DataFrame({
# # #                 'UserEmail': [email],
# # #                 'Embedding': [zero_embedding_encoded],
# # #                 'LastRecommendedIndex': [0],
# # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # #             })
# # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # #             save_user_embeddings(user_embeddings)
# # #             print(f"Initialized zero embedding for user {email}.")
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# # #     return jsonify({'message': 'User registered successfully.'}), 201

# # # @app.route('/login', methods=['POST'])
# # # def login():
# # #     data = request.get_json()
# # #     email = data.get('email')
# # #     password = data.get('password')

# # #     if not all([email, password]):
# # #         return jsonify({'message': 'Email and password are required.'}), 400

# # #     try:
# # #         users = load_users()
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# # #     user = users[users['Email'] == email]

# # #     if user.empty:
# # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # #     stored_password = user.iloc[0]['Password']
# # #     full_name = user.iloc[0]['FullName']

# # #     if not check_password_hash(stored_password, password):
# # #         return jsonify({'message': 'Invalid email or password.'}), 401

# # #     try:
# # #         token = jwt.encode({
# # #             'email': email,
# # #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# # #         }, app.config['SECRET_KEY'], algorithm="HS256")
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# # #     # Ensure user has an embedding; initialize if not
# # #     try:
# # #         user_embeddings = load_user_embeddings()
# # #         if email not in user_embeddings['UserEmail'].values:
# # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# # #             zero_embedding = np.zeros(embedding_dim)
# # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # #             new_embedding = pd.DataFrame({
# # #                 'UserEmail': [email],
# # #                 'Embedding': [zero_embedding_encoded],
# # #                 'LastRecommendedIndex': [0],
# # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # #             })
# # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # #             save_user_embeddings(user_embeddings)
# # #             print(f"Initialized zero embedding for user {email} on login.")
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# # #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # # @app.route('/protected', methods=['GET'])
# # # @token_required
# # # def protected_route(current_user_email):
# # #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # # --- Removed the /get-images Endpoint ---
# # # # Since we are now using only /recommend-images, we remove /get-images.
# # # # Uncomment the following lines if you want to retain /get-images for any reason.

# # # # @app.route('/get-images', methods=['GET'])
# # # # @token_required
# # # # def get_images(current_user_email):
# # # #     """
# # # #     Fetch a batch of images for the user.
# # # #     If the user has not liked any images, return random indices.
# # # #     Otherwise, fetch based on recommendations.
# # # #     """
# # # #     # ... existing code ...
# # # #     pass  # This endpoint has been removed as per requirements.

# # # # --- Modified /like-image Endpoint ---
# # # @app.route('/like-image', methods=['POST'])
# # # @token_required
# # # def like_image(current_user_email):
# # #     """
# # #     Records a user's like for an image and updates embeddings immediately.
# # #     Also updates the image_likes.xlsx file to track which users liked which images.
# # #     """
# # #     data = request.get_json()
# # #     image_index = data.get('image_index')

# # #     if image_index is None:
# # #         return jsonify({'message': 'Image index is required.'}), 400

# # #     # Ensure image_index is int
# # #     try:
# # #         image_index = int(image_index)
# # #     except ValueError:
# # #         return jsonify({'message': 'Image index must be an integer.'}), 400

# # #     # Record the like
# # #     try:
# # #         user_likes = load_user_likes()
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # #     # Ensure image_index is within range
# # #     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
# # #         return jsonify({'message': 'Invalid image index.'}), 400

# # #     new_like = pd.DataFrame({
# # #         'UserEmail': [current_user_email],
# # #         'ImageIndex': [image_index],
# # #         'LikedAt': [datetime.datetime.utcnow()]
# # #     })

# # #     try:
# # #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# # #         save_user_likes(user_likes)
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# # #     # --- Update image_likes.xlsx ---
# # #     try:
# # #         image_likes = load_image_likes()
# # #         if image_likes['ImageIndex'].dtype != int and image_likes['ImageIndex'].dtype != np.int64:
# # #             image_likes['ImageIndex'] = image_likes['ImageIndex'].astype(int)
# # #         if image_index in image_likes['ImageIndex'].values:
# # #             # Get the current list of users who liked this image
# # #             users = image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'].iloc[0]
# # #             if current_user_email not in users:
# # #                 users.append(current_user_email)
# # #                 image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'] = [users]
# # #                 save_image_likes(image_likes)
# # #                 print(f"Added user {current_user_email} to ImageIndex {image_index} likes.")
# # #             else:
# # #                 print(f"User {current_user_email} already liked ImageIndex {image_index}.")
# # #         else:
# # #             # If the image index is not present, add it
# # #             image_likes = image_likes.append({'ImageIndex': image_index, 'Users': [current_user_email]}, ignore_index=True)
# # #             save_image_likes(image_likes)
# # #             print(f"Initialized likes for ImageIndex {image_index} with user {current_user_email}.")
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error updating image likes: {str(e)}'}), 500

# # #     # --- Update user embedding immediately after each like ---
# # #     try:
# # #         user_embeddings = load_user_embeddings()
# # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # #         if user_embedding_row.empty:
# # #             # Initialize embedding with zeros if not found
# # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # #             zero_embedding = np.zeros(embedding_dim)
# # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # #             new_user_embedding = pd.DataFrame({
# # #                 'UserEmail': [current_user_email],
# # #                 'Embedding': [zero_embedding_encoded],
# # #                 'LastRecommendedIndex': [0],
# # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # #             })
# # #             user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# # #             save_user_embeddings(user_embeddings)
# # #             user_embedding = zero_embedding
# # #             print(f"Initialized zero embedding for user {current_user_email} during like update.")
# # #         else:
# # #             # Decode the existing embedding
# # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# # #         # Fetch the liked image embedding
# # #         if all_embeddings is not None:
# # #             liked_embedding = all_embeddings[image_index]
# # #             if np.linalg.norm(liked_embedding) != 0:
# # #                 liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
# # #             else:
# # #                 liked_embedding = liked_embedding
# # #         else:
# # #             # If embeddings are not available, use zero embedding
# # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # #             liked_embedding = np.zeros(embedding_dim)

# # #         # Combine with previous embedding using the specified weight
# # #         weight = 0.7  # Define the weight as needed
# # #         combined_embedding = combine_embeddings_for_recommendation(
# # #             current_embedding=liked_embedding,
# # #             previous_embedding=user_embedding,
# # #             weight=weight
# # #         )
# # #         norm = np.linalg.norm(combined_embedding)
# # #         if norm != 0:
# # #             combined_embedding = combined_embedding / norm
# # #         else:
# # #             combined_embedding = combined_embedding

# # #         # Update the embedding in the dataframe
# # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# # #         # Reset LastRecommendedIndex since embedding has been updated
# # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# # #         # Update LastEmbeddingUpdate timestamp
# # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# # #         # Save the updated embeddings
# # #         save_user_embeddings(user_embeddings)
# # #         print(f"Updated embedding for user {current_user_email} after like.")

# # #     except Exception as e:
# # #         return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# # #     return jsonify({'message': 'Image liked successfully.'}), 200

# # # @app.route('/recommend-images', methods=['GET'])
# # # @token_required
# # # def recommend_images(current_user_email):
# # #     """
# # #     Provides personalized recommendations based on user embeddings.
# # #     """
# # #     try:
# # #         user_embeddings = load_user_embeddings()
# # #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# # #         if user_embedding_row.empty:
# # #             # Initialize embedding with zeros if not found
# # #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# # #             zero_embedding = np.zeros(embedding_dim)
# # #             zero_embedding_encoded = encode_embedding(zero_embedding)
# # #             new_embedding = pd.DataFrame({
# # #                 'UserEmail': [current_user_email],
# # #                 'Embedding': [zero_embedding_encoded],
# # #                 'LastRecommendedIndex': [0],
# # #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# # #             })
# # #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# # #             save_user_embeddings(user_embeddings)
# # #             user_embedding = zero_embedding.reshape(1, -1)
# # #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# # #         else:
# # #             # Decode the existing embedding
# # #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# # #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# # #     # Check if user has liked any images
# # #     try:
# # #         user_likes = load_user_likes()
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# # #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # #     if not user_liked_images:
# # #         # User hasn't liked any images yet, return random 40 images
# # #         if train_data is not None:
# # #             num_images = len(train_data)
# # #             sample_size = 40 if num_images >= 40 else num_images
# # #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# # #         else:
# # #             return jsonify({'message': 'No images available.'}), 500
# # #     else:
# # #         if all_embeddings is None:
# # #             return jsonify({'message': 'Embeddings not available.'}), 500

# # #         # Ensure user_embedding has the correct dimension
# # #         embedding_dim = all_embeddings.shape[1]
# # #         if user_embedding.shape[1] != embedding_dim:
# # #             if user_embedding.shape[1] > embedding_dim:
# # #                 user_embedding = user_embedding[:, :embedding_dim]
# # #                 print("Trimmed user_embedding to match embedding_dim.")
# # #             else:
# # #                 padding_size = embedding_dim - user_embedding.shape[1]
# # #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# # #                 user_embedding = np.hstack((user_embedding, padding))
# # #                 print(f"Padded user_embedding with {padding_size} zeros.")
# # #             # Update the embedding in the dataframe
# # #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# # #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# # #             save_user_embeddings(user_embeddings)

# # #         # Compute similarities
# # #         similarities = cosine_similarity(user_embedding, all_embeddings)
# # #         top_indices = similarities.argsort()[0][::-1]

# # #         # Exclude already liked images
# # #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# # #         # Fetch LastRecommendedIndex
# # #         try:
# # #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# # #         except:
# # #             last_recommended_index = 0

# # #         # Define batch size
# # #         batch_size = 40  # Fetch top 40

# # #         # Select the next batch
# # #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# # #         # Update LastRecommendedIndex
# # #         new_last_recommended_index = last_recommended_index + batch_size
# # #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# # #         save_user_embeddings(user_embeddings)

# # #     recommendations = []

# # #     for idx in indices:
# # #         try:
# # #             artwork = train_data[int(idx)]  # Convert to int
# # #         except IndexError:
# # #             print(f"Index {idx} is out of bounds for the dataset.")
# # #             continue
# # #         except TypeError as te:
# # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # #             continue

# # #         curr_metadata = {
# # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # #             "style": artwork.get('style', 'Unknown Style'),
# # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # #             "description": artwork.get('description', 'No Description Available')
# # #         }

# # #         image_data_or_url = artwork.get('image', None)

# # #         if isinstance(image_data_or_url, str):
# # #             try:
# # #                 response = requests.get(image_data_or_url)
# # #                 if response.status_code == 200:
# # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # #                 else:
# # #                     artwork_image = None
# # #             except Exception as e:
# # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # #                 artwork_image = None
# # #         elif isinstance(image_data_or_url, Image.Image):
# # #             artwork_image = image_data_or_url
# # #         else:
# # #             artwork_image = None

# # #         if artwork_image:
# # #             try:
# # #                 img_base64 = encode_image_to_base64(artwork_image)
# # #             except Exception as e:
# # #                 print(f"Error encoding image to base64: {e}")
# # #                 img_base64 = None
# # #         else:
# # #             img_base64 = None

# # #         recommendations.append({
# # #             'index': int(idx),  # Convert to int
# # #             'artist': curr_metadata['artist'],
# # #             'style': curr_metadata['style'],
# # #             'genre': curr_metadata['genre'],
# # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # #             'image': img_base64
# # #         })

# # #     return jsonify({'images': recommendations}), 200

# # # @app.route('/chat', methods=['POST'])
# # # @token_required
# # # def chat(current_user_email):
# # #     """
# # #     Handle chat requests with text and optional image.
# # #     Processes the inputs and returns a response.
# # #     """
# # #     text = request.form.get('text', '').strip()
# # #     image_file = request.files.get('image', None)

# # #     image_data = None
# # #     if image_file:
# # #         try:
# # #             image_bytes = image_file.read()
# # #             image = Image.open(io.BytesIO(image_bytes))
# # #             image = image.convert('RGB')
# # #             image_data = image
# # #         except Exception as e:
# # #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# # #     try:
# # #         result = predict(text, image_data)
# # #         return jsonify(result), 200
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # # def predict(text, image_data=None):
# # #     """
# # #     Process the input text and image, generate recommendations,
# # #     and return them with explanations and metadata.
# # #     """
# # #     if not all([
# # #         all_embeddings is not None, 
# # #         train_data is not None, 
# # #         clip_model is not None, 
# # #         clip_processor is not None, 
# # #         blip_model is not None, 
# # #         blip_processor is not None
# # #     ]):
# # #         return {'message': 'Server not fully initialized. Please check the logs.'}

# # #     input_image = image_data
# # #     user_text = text

# # #     if input_image:
# # #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# # #         print(f"Generated image caption: {image_caption}")
# # #     else:
# # #         image_caption = ""

# # #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# # #     print(f"Context-aware text: {context_aware_text}")

# # #     if input_image:
# # #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# # #     else:
# # #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# # #     inputs = {key: value.to(device) for key, value in inputs.items()}
# # #     print("Preprocessed inputs for CLIP.")

# # #     with torch.no_grad():
# # #         if input_image:
# # #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# # #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# # #             image_features_np = image_features.cpu().detach().numpy()
# # #         else:
# # #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# # #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# # #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# # #         text_features_np = text_features.cpu().detach().numpy()
# # #     print("Generated and normalized image and text features using CLIP.")

# # #     weight_img = 0.1
# # #     weight_text = 0.9

# # #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# # #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # #     if norm != 0:
# # #         final_embedding = final_embedding / norm
# # #     else:
# # #         final_embedding = final_embedding
# # #     print("Computed final combined embedding.")

# # #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# # #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# # #     embedding_dim = all_embeddings.shape[1]
# # #     if final_embedding.shape[1] != embedding_dim:
# # #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# # #         if final_embedding.shape[1] > embedding_dim:
# # #             final_embedding = final_embedding[:, :embedding_dim]
# # #             print("Trimmed final_embedding.")
# # #         else:
# # #             padding_size = embedding_dim - final_embedding.shape[1]
# # #             padding = np.zeros((final_embedding.shape[0], padding_size))
# # #             final_embedding = np.hstack((final_embedding, padding))
# # #             print(f"Padded final_embedding with {padding_size} zeros.")
# # #         # Update the embedding in the dataframe
# # #         final_embedding_normalized = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# # #         # Note: Since this is within predict, not updating user embeddings
# # #         # If needed, update here or elsewhere
# # #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# # #     similarities = cosine_similarity(final_embedding, all_embeddings)
# # #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# # #     top_n = 10
# # #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# # #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# # #     recommended_artworks = [int(i) for i in top_n_indices]

# # #     recommendations = []

# # #     for rank, i in enumerate(recommended_artworks, start=1):
# # #         try:
# # #             artwork = train_data[int(i)]
# # #         except IndexError:
# # #             print(f"Index {i} is out of bounds for the dataset.")
# # #             continue

# # #         curr_metadata = {
# # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # #             "style": artwork.get('style', 'Unknown Style'),
# # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # #             "description": artwork.get('description', 'No Description Available')
# # #         }

# # #         image_data_or_url = artwork.get('image', None)

# # #         if isinstance(image_data_or_url, str):
# # #             try:
# # #                 response = requests.get(image_data_or_url)
# # #                 if response.status_code == 200:
# # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # #                 else:
# # #                     artwork_image = None
# # #             except Exception as e:
# # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # #                 artwork_image = None
# # #         elif isinstance(image_data_or_url, Image.Image):
# # #             artwork_image = image_data_or_url
# # #         else:
# # #             artwork_image = None

# # #         if artwork_image:
# # #             try:
# # #                 img_base64 = encode_image_to_base64(artwork_image)
# # #             except Exception as e:
# # #                 print(f"Error encoding image to base64: {e}")
# # #                 img_base64 = None
# # #         else:
# # #             img_base64 = None

# # #         recommendations.append({
# # #             'rank': rank,
# # #             'index': int(i),  # Convert to int
# # #             'artist': curr_metadata['artist'],
# # #             'style': curr_metadata['style'],
# # #             'genre': curr_metadata['genre'],
# # #             # 'description': curr_metadata['description'],  # Optional: Uncomment if needed
# # #             'image': img_base64
# # #         })

# # #     response_text = "Here are the recommended artworks based on your preferences:"

# # #     return {
# # #         'response': response_text,
# # #         'recommendations': recommendations
# # #     }

# # # @app.route('/trending', methods=['GET'])
# # # @token_required
# # # def trending(current_user_email):
# # #     """
# # #     Retrieves the top 40 trending images based on the number of likes.
# # #     Returns the images along with their like counts.
# # #     """
# # #     try:
# # #         image_likes = load_image_likes()
# # #     except Exception as e:
# # #         return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

# # #     # Calculate like counts for each image
# # #     image_likes['LikeCount'] = image_likes['Users'].apply(len)

# # #     # Sort images by LikeCount descendingly
# # #     top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

# # #     recommendations = []

# # #     for _, row in top_images.iterrows():
# # #         idx = row['ImageIndex']
# # #         like_count = row['LikeCount']

# # #         try:
# # #             artwork = train_data[int(idx)]
# # #         except IndexError:
# # #             print(f"Index {idx} is out of bounds for the dataset.")
# # #             continue
# # #         except TypeError as te:
# # #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# # #             continue

# # #         curr_metadata = {
# # #             "artist": artwork.get('artist', 'Unknown Artist'),
# # #             "style": artwork.get('style', 'Unknown Style'),
# # #             "genre": artwork.get('genre', 'Unknown Genre'),
# # #             "description": artwork.get('description', 'No Description Available')
# # #         }

# # #         image_data_or_url = artwork.get('image', None)

# # #         if isinstance(image_data_or_url, str):
# # #             try:
# # #                 response = requests.get(image_data_or_url)
# # #                 if response.status_code == 200:
# # #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # #                 else:
# # #                     artwork_image = None
# # #             except Exception as e:
# # #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# # #                 artwork_image = None
# # #         elif isinstance(image_data_or_url, Image.Image):
# # #             artwork_image = image_data_or_url
# # #         else:
# # #             artwork_image = None

# # #         if artwork_image:
# # #             try:
# # #                 img_base64 = encode_image_to_base64(artwork_image)
# # #             except Exception as e:
# # #                 print(f"Error encoding image to base64: {e}")
# # #                 img_base64 = None
# # #         else:
# # #             img_base64 = None

# # #         recommendations.append({
# # #             'index': int(idx),  # Convert to int
# # #             'artist': curr_metadata['artist'],
# # #             'style': curr_metadata['style'],
# # #             'genre': curr_metadata['genre'],
# # #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # #             'image': img_base64,
# # #             'like_count': like_count
# # #         })

# # #     return jsonify({'trending_images': recommendations}), 200

# # # @app.route('/get_all_liked', methods=['GET'])
# # # @token_required
# # # def get_all_liked(current_user_email):
# # #     """
# # #     Retrieves all liked images for the authenticated user.
# # #     """
# # #     try:
# # #         # Load user likes
# # #         user_likes = load_user_likes()
# # #         liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# # #         if not liked_image_indices:
# # #             return jsonify({'liked_images': []}), 200

# # #         # Fetch image data from train_data
# # #         liked_images = []
# # #         for idx in liked_image_indices:
# # #             try:
# # #                 artwork = train_data[int(idx)]
# # #             except IndexError:
# # #                 print(f"Index {idx} is out of bounds for the dataset.")
# # #                 continue
# # #             except TypeError as te:
# # #                 print(f"TypeError accessing train_data with idx={idx}: {te}")
# # #                 continue

# # #             curr_metadata = {
# # #                 "artist": artwork.get('artist', 'Unknown Artist'),
# # #                 "style": artwork.get('style', 'Unknown Style'),
# # #                 "genre": artwork.get('genre', 'Unknown Genre'),
# # #                 "description": artwork.get('description', 'No Description Available')
# # #             }

# # #             image_data_or_url = artwork.get('image', None)

# # #             if isinstance(image_data_or_url, str):
# # #                 try:
# # #                     response = requests.get(image_data_or_url)
# # #                     if response.status_code == 200:
# # #                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# # #                     else:
# # #                         artwork_image = None
# # #                 except Exception as e:
# # #                     print(f"Error fetching image from {image_data_or_url}: {e}")
# # #                     artwork_image = None
# # #             elif isinstance(image_data_or_url, Image.Image):
# # #                 artwork_image = image_data_or_url
# # #             else:
# # #                 artwork_image = None

# # #             if artwork_image:
# # #                 try:
# # #                     img_base64 = encode_image_to_base64(artwork_image)
# # #                 except Exception as e:
# # #                     print(f"Error encoding image to base64: {e}")
# # #                     img_base64 = None
# # #             else:
# # #                 img_base64 = None

# # #             # Fetch the timestamp of the like
# # #             like_timestamp = user_likes[(user_likes['UserEmail'] == current_user_email) & (user_likes['ImageIndex'] == idx)]['LikedAt'].iloc[0]
# # #             like_timestamp_iso = like_timestamp.isoformat()

# # #             liked_images.append({
# # #                 'index': int(idx),  # Convert to int
# # #                 'artist': curr_metadata['artist'],
# # #                 'style': curr_metadata['style'],
# # #                 'genre': curr_metadata['genre'],
# # #                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# # #                 'image': img_base64,
# # #                 'timestamp': like_timestamp_iso  # ISO formatted timestamp
# # #             })

# # #         return jsonify({'liked_images': liked_images}), 200

# # #     except Exception as e:
# # #         return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500

# # # # --- Additional Initialization for image_likes.xlsx ---
# # # # Initialize image_likes.xlsx entries for all images if not already present
# # # try:
# # #     if all_embeddings is not None and train_data is not None:
# # #         image_likes = load_image_likes()
# # #         existing_indices = set(image_likes['ImageIndex'].tolist())
# # #         all_indices = set(range(all_embeddings.shape[0]))
# # #         missing_indices = all_indices - existing_indices

# # #         if missing_indices:
# # #             new_entries = pd.DataFrame({
# # #                 'ImageIndex': list(missing_indices),
# # #                 'Users': [json.dumps([]) for _ in range(len(missing_indices))]  # Initialize with empty lists
# # #             })
# # #             image_likes = pd.concat([image_likes, new_entries], ignore_index=True)
# # #             save_image_likes(image_likes)
# # #             print(f"Initialized likes for {len(missing_indices)} images.")
# # #         else:
# # #             print("All images already have like entries.")
# # #     else:
# # #         print("Embeddings or training data not available. Skipping image likes initialization.")
# # # except Exception as e:
# # #     print(f"Error initializing image likes: {e}")

# # # if __name__ == '__main__':
# # #     app.run(debug=True)


# # # backend/app.py

# # from flask import Flask, request, jsonify
# # import pandas as pd
# # import os
# # from werkzeug.security import generate_password_hash, check_password_hash
# # import jwt
# # import datetime
# # from functools import wraps
# # from flask_cors import CORS
# # from dotenv import load_dotenv
# # import io
# # from PIL import Image
# # import numpy as np
# # import torch
# # from transformers import CLIPProcessor, CLIPModel
# # from transformers import BlipProcessor, BlipForConditionalGeneration
# # from datasets import load_dataset
# # from sklearn.metrics.pairwise import cosine_similarity
# # import requests
# # import base64
# # import json
# # from filelock import FileLock, Timeout

# # def display_image(image_data):
# #     # Function to display images (not used in backend)
# #     pass

# # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# #     with torch.no_grad():
# #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# #     return caption

# # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# #     margin = 0.05
# #     if sim_image > sim_text + margin:
# #         reason = "the style and composition of the input image."
# #     elif sim_text > sim_image + margin:
# #         reason = "your textual preferences for nature and the specified colors."
# #     else:
# #         reason = "a balanced combination of both your image and textual preferences."

# #     explanation = (
# #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# #         f"is recommended {reason} "
# #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# #     )
# #     return explanation

# # def encode_image_to_base64(image):
# #     buffered = io.BytesIO()
# #     image.save(buffered, format="JPEG")
# #     img_bytes = buffered.getvalue()
# #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# #     return img_base64

# # def decode_embedding(embedding_str):
# #     return np.array(json.loads(embedding_str))

# # def encode_embedding(embedding_array):
# #     return json.dumps(embedding_array.tolist())

# # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# #     """
# #     Combines the current embedding with the previous one using a weighted average.
# #     """
# #     if previous_embedding is None:
# #         return current_embedding
# #     return weight * current_embedding + (1 - weight) * previous_embedding

# # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# #     """
# #     Recommends the top-k similar artworks based on cosine similarity.
# #     """
# #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# #     return top_k_indices

# # # Determine the device to use for computation
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(f"Using device: {device}")

# # # Load combined embeddings
# # try:
# #     all_embeddings = np.load('combined_embeddings.npy')
# #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # except FileNotFoundError:
# #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# #     all_embeddings = None

# # if all_embeddings is not None:
# #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# #     print("Normalized all_embeddings for cosine similarity.")
# # else:
# #     print("Skipping normalization due to missing embeddings.")

# # # Load the dataset (e.g., WikiArt for training data)
# # try:
# #     ds = load_dataset("Artificio/WikiArt")
# #     train_data = ds['train']
# #     print("Loaded WikiArt dataset successfully.")
# # except Exception as e:
# #     print(f"Error loading dataset: {e}")
# #     train_data = None

# # # Load CLIP model and processor
# # try:
# #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# #     clip_model.to(device)
# #     print("Loaded CLIP model and processor successfully.")
# # except Exception as e:
# #     print(f"Error loading CLIP model: {e}")
# #     clip_model = None
# #     clip_processor = None

# # # Load BLIP model and processor for image captioning
# # try:
# #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# #     blip_model.to(device)
# #     print("Loaded BLIP model and processor successfully.")
# # except Exception as e:
# #     print(f"Error loading BLIP model: {e}")
# #     blip_model = None
# #     blip_processor = None

# # # Load environment variables from .env file
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Retrieve the secret key from environment variables
# # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # Ensure that the secret key is set
# # if not app.config['SECRET_KEY']:
# #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # Define Excel database files
# # DATABASE_USERS = 'users.xlsx'
# # DATABASE_LIKES = 'user_likes.xlsx'
# # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'
# # DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# # # Initialize Excel files if they don't exist
# # for db_file, columns in [
# #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate']),
# #     (DATABASE_IMAGE_LIKES, ['ImageIndex', 'Users'])  # 'Users' will store JSON strings
# # ]:
# #     if not os.path.exists(db_file):
# #         try:
# #             if db_file == DATABASE_IMAGE_LIKES:
# #                 df = pd.DataFrame(columns=columns)
# #             else:
# #                 df = pd.DataFrame(columns=columns)
# #             df.to_excel(db_file, index=False, engine='openpyxl')
# #             print(f"Created {db_file} with columns: {columns}")
# #         except Exception as e:
# #             print(f"Error creating {db_file}: {e}")

# # def load_users():
# #     lock_path = DATABASE_USERS + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             return pd.read_excel(DATABASE_USERS, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for loading users.")
# #     except Exception as e:
# #         # Handle decompression errors or other read errors
# #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# #             print(f"Corrupted {DATABASE_USERS}. Recreating it.")
# #             df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# #             try:
# #                 df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# #                 print(f"Recreated {DATABASE_USERS}.")
# #             except Exception as ex:
# #                 print(f"Failed to recreate {DATABASE_USERS}: {ex}")
# #                 raise Exception(f"Failed to recreate {DATABASE_USERS}: {ex}")
# #             return df
# #         else:
# #             raise e

# # def save_users(df):
# #     lock_path = DATABASE_USERS + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for saving users.")
# #     except Exception as e:
# #         raise e

# # def load_user_likes():
# #     lock_path = DATABASE_LIKES + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             return pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for loading user likes.")
# #     except Exception as e:
# #         # Handle decompression errors
# #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# #             print(f"Corrupted {DATABASE_LIKES}. Recreating it.")
# #             df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# #             try:
# #                 df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# #                 print(f"Recreated {DATABASE_LIKES}.")
# #             except Exception as ex:
# #                 print(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# #                 raise Exception(f"Failed to recreate {DATABASE_LIKES}: {ex}")
# #             return df
# #         else:
# #             raise e

# # def save_user_likes(df):
# #     lock_path = DATABASE_LIKES + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for saving user likes.")
# #     except Exception as e:
# #         raise e

# # def load_user_embeddings():
# #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             return pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for loading user embeddings.")
# #     except Exception as e:
# #         # Handle decompression errors
# #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# #             print(f"Corrupted {DATABASE_EMBEDDINGS}. Recreating it.")
# #             df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# #             try:
# #                 df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# #                 print(f"Recreated {DATABASE_EMBEDDINGS}.")
# #             except Exception as ex:
# #                 print(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# #                 raise Exception(f"Failed to recreate {DATABASE_EMBEDDINGS}: {ex}")
# #             return df
# #         else:
# #             raise e

# # def save_user_embeddings(df):
# #     lock_path = DATABASE_EMBEDDINGS + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for saving user embeddings.")
# #     except Exception as e:
# #         raise e

# # def load_image_likes():
# #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         with lock.acquire(timeout=10):
# #             df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
# #             # Ensure 'Users' column is parsed from JSON strings to lists
# #             def parse_users(x):
# #                 if isinstance(x, str):
# #                     try:
# #                         return json.loads(x)
# #                     except json.JSONDecodeError:
# #                         return []
# #                 elif isinstance(x, list):
# #                     return x
# #                 else:
# #                     return []

# #             df['Users'] = df['Users'].apply(parse_users)
# #             return df
# #     except Timeout:
# #         raise Exception("Could not acquire lock for loading image likes.")
# #     except Exception as e:
# #         # Handle decompression errors or other read errors
# #         if 'decompressing data' in str(e) or 'not a valid zip file' in str(e):
# #             print(f"Corrupted {DATABASE_IMAGE_LIKES}. Recreating it.")
# #             df = pd.DataFrame(columns=['ImageIndex', 'Users'])
# #             try:
# #                 df.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# #                 print(f"Recreated {DATABASE_IMAGE_LIKES}.")
# #             except Exception as ex:
# #                 print(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# #                 raise Exception(f"Failed to recreate {DATABASE_IMAGE_LIKES}: {ex}")
# #             return df
# #         else:
# #             raise e

# # def save_image_likes(df):
# #     lock_path = DATABASE_IMAGE_LIKES + '.lock'
# #     lock = FileLock(lock_path)
# #     try:
# #         # Convert 'Users' lists to JSON strings before saving
# #         df_copy = df.copy()
# #         df_copy['Users'] = df_copy['Users'].apply(lambda x: json.dumps(x))
# #         with lock.acquire(timeout=10):
# #             df_copy.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# #     except Timeout:
# #         raise Exception("Could not acquire lock for saving image likes.")
# #     except Exception as e:
# #         raise e

# # def token_required(f):
# #     @wraps(f)
# #     def decorated(*args, **kwargs):
# #         token = None

# #         if 'Authorization' in request.headers:
# #             auth_header = request.headers['Authorization']
# #             try:
# #                 token = auth_header.split(" ")[1]
# #             except IndexError:
# #                 return jsonify({'message': 'Token format invalid!'}), 401

# #         if not token:
# #             return jsonify({'message': 'Token is missing!'}), 401

# #         try:
# #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# #             current_user_email = data['email']
# #         except jwt.ExpiredSignatureError:
# #             return jsonify({'message': 'Token has expired!'}), 401
# #         except jwt.InvalidTokenError:
# #             return jsonify({'message': 'Invalid token!'}), 401

# #         try:
# #             users = load_users()
# #         except Exception as e:
# #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #         user = users[users['Email'] == current_user_email]
# #         if user.empty:
# #             return jsonify({'message': 'User not found!'}), 401

# #         return f(current_user_email, *args, **kwargs)

# #     return decorated

# # @app.route('/signup', methods=['POST'])
# # def signup():
# #     data = request.get_json()
# #     full_name = data.get('full_name')
# #     email = data.get('email')
# #     password = data.get('password')

# #     if not all([full_name, email, password]):
# #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# #     try:
# #         users = load_users()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #     if email in users['Email'].values:
# #         return jsonify({'message': 'Email already exists.'}), 400

# #     hashed_password = generate_password_hash(password)

# #     new_user = pd.DataFrame({
# #         'FullName': [full_name],
# #         'Email': [email],
# #         'Password': [hashed_password]
# #     })

# #     try:
# #         users = pd.concat([users, new_user], ignore_index=True)
# #     except Exception as e:
# #         return jsonify({'message': f'Error appending new user: {str(e)}'}), 500

# #     try:
# #         save_users(users)
# #     except Exception as e:
# #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         if email not in user_embeddings['UserEmail'].values:
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             print(f"Initialized zero embedding for user {email}.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# #     return jsonify({'message': 'User registered successfully.'}), 201

# # @app.route('/login', methods=['POST'])
# # def login():
# #     data = request.get_json()
# #     email = data.get('email')
# #     password = data.get('password')

# #     if not all([email, password]):
# #         return jsonify({'message': 'Email and password are required.'}), 400

# #     try:
# #         users = load_users()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #     user = users[users['Email'] == email]

# #     if user.empty:
# #         return jsonify({'message': 'Invalid email or password.'}), 401

# #     stored_password = user.iloc[0]['Password']
# #     full_name = user.iloc[0]['FullName']

# #     if not check_password_hash(stored_password, password):
# #         return jsonify({'message': 'Invalid email or password.'}), 401

# #     try:
# #         token = jwt.encode({
# #             'email': email,
# #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# #         }, app.config['SECRET_KEY'], algorithm="HS256")
# #     except Exception as e:
# #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# #     # Ensure user has an embedding; initialize if not
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         if email not in user_embeddings['UserEmail'].values:
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             print(f"Initialized zero embedding for user {email} on login.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # @app.route('/protected', methods=['GET'])
# # @token_required
# # def protected_route(current_user_email):
# #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # --- Removed the /get-images Endpoint ---
# # # Since we are now using only /recommend-images, we remove /get-images.
# # # Uncomment the following lines if you want to retain /get-images for any reason.

# # # @app.route('/get-images', methods=['GET'])
# # # @token_required
# # # def get_images(current_user_email):
# # #     """
# # #     Fetch a batch of images for the user.
# # #     If the user has not liked any images, return random indices.
# # #     Otherwise, fetch based on recommendations.
# # #     """
# # #     # ... existing code ...
# # #     pass  # This endpoint has been removed as per requirements.

# # # --- Modified /like-image Endpoint ---
# # @app.route('/like-image', methods=['POST'])
# # @token_required
# # def like_image(current_user_email):
# #     """
# #     Records a user's like for an image and updates embeddings immediately.
# #     Also updates the image_likes.xlsx file to track which users liked which images.
# #     """
# #     data = request.get_json()
# #     image_index = data.get('image_index')

# #     if image_index is None:
# #         return jsonify({'message': 'Image index is required.'}), 400

# #     # Ensure image_index is int
# #     try:
# #         image_index = int(image_index)
# #     except ValueError:
# #         return jsonify({'message': 'Image index must be an integer.'}), 400

# #     # Record the like
# #     try:
# #         user_likes = load_user_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# #     # Ensure image_index is within range
# #     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
# #         return jsonify({'message': 'Invalid image index.'}), 400

# #     new_like = pd.DataFrame({
# #         'UserEmail': [current_user_email],
# #         'ImageIndex': [image_index],
# #         'LikedAt': [datetime.datetime.utcnow()]
# #     })

# #     try:
# #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# #         save_user_likes(user_likes)
# #     except Exception as e:
# #         return jsonify({'message': f'Error saving like: {str(e)}'}), 500

# #     # --- Update image_likes.xlsx ---
# #     try:
# #         image_likes = load_image_likes()
# #         if image_index in image_likes['ImageIndex'].values:
# #             # Get the current list of users who liked this image
# #             users = image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'].iloc[0]
# #             if not isinstance(users, list):
# #                 users = []
# #             if current_user_email not in users:
# #                 users.append(current_user_email)
# #                 image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'] = [users]
# #                 save_image_likes(image_likes)
# #                 print(f"Added user {current_user_email} to ImageIndex {image_index} likes.")
# #             else:
# #                 print(f"User {current_user_email} already liked ImageIndex {image_index}.")
# #         else:
# #             # If the image index is not present, add it
# #             image_likes = image_likes.append({'ImageIndex': image_index, 'Users': [current_user_email]}, ignore_index=True)
# #             save_image_likes(image_likes)
# #             print(f"Initialized likes for ImageIndex {image_index} with user {current_user_email}.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error updating image likes: {str(e)}'}), 500

# #     # --- Update user embedding immediately after each like ---
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# #         if user_embedding_row.empty:
# #             # Initialize embedding with zeros if not found
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_user_embedding = pd.DataFrame({
# #                 'UserEmail': [current_user_email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             user_embedding = zero_embedding
# #             print(f"Initialized zero embedding for user {current_user_email} during like update.")
# #         else:
# #             # Decode the existing embedding
# #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# #         # Fetch the liked image embedding
# #         if all_embeddings is not None:
# #             liked_embedding = all_embeddings[image_index]
# #             if np.linalg.norm(liked_embedding) != 0:
# #                 liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
# #             else:
# #                 liked_embedding = liked_embedding
# #         else:
# #             # If embeddings are not available, use zero embedding
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             liked_embedding = np.zeros(embedding_dim)

# #         # Combine with previous embedding using the specified weight
# #         weight = 0.7  # Define the weight as needed
# #         combined_embedding = combine_embeddings_for_recommendation(
# #             current_embedding=liked_embedding,
# #             previous_embedding=user_embedding,
# #             weight=weight
# #         )
# #         norm = np.linalg.norm(combined_embedding)
# #         if norm != 0:
# #             combined_embedding = combined_embedding / norm
# #         else:
# #             combined_embedding = combined_embedding

# #         # Update the embedding in the dataframe
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# #         # Reset LastRecommendedIndex since embedding has been updated
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# #         # Update LastEmbeddingUpdate timestamp
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# #         # Save the updated embeddings
# #         save_user_embeddings(user_embeddings)
# #         print(f"Updated embedding for user {current_user_email} after like.")

# #     except Exception as e:
# #         return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# #     return jsonify({'message': 'Image liked successfully.'}), 200

# # @app.route('/recommend-images', methods=['GET'])
# # @token_required
# # def recommend_images(current_user_email):
# #     """
# #     Provides personalized recommendations based on user embeddings.
# #     """
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# #         if user_embedding_row.empty:
# #             # Initialize embedding with zeros if not found
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [current_user_email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             user_embedding = zero_embedding.reshape(1, -1)
# #             print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# #         else:
# #             # Decode the existing embedding
# #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# #             last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# #     # Check if user has liked any images
# #     try:
# #         user_likes = load_user_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# #     user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# #     if not user_liked_images:
# #         # User hasn't liked any images yet, return random 40 images
# #         if train_data is not None:
# #             num_images = len(train_data)
# #             sample_size = 40 if num_images >= 40 else num_images
# #             indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# #         else:
# #             return jsonify({'message': 'No images available.'}), 500
# #     else:
# #         if all_embeddings is None:
# #             return jsonify({'message': 'Embeddings not available.'}), 500

# #         # Ensure user_embedding has the correct dimension
# #         embedding_dim = all_embeddings.shape[1]
# #         if user_embedding.shape[1] != embedding_dim:
# #             if user_embedding.shape[1] > embedding_dim:
# #                 user_embedding = user_embedding[:, :embedding_dim]
# #                 print("Trimmed user_embedding to match embedding_dim.")
# #             else:
# #                 padding_size = embedding_dim - user_embedding.shape[1]
# #                 padding = np.zeros((user_embedding.shape[0], padding_size))
# #                 user_embedding = np.hstack((user_embedding, padding))
# #                 print(f"Padded user_embedding with {padding_size} zeros.")
# #             # Update the embedding in the dataframe
# #             user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# #             save_user_embeddings(user_embeddings)

# #         # Compute similarities
# #         similarities = cosine_similarity(user_embedding, all_embeddings)
# #         top_indices = similarities.argsort()[0][::-1]

# #         # Exclude already liked images
# #         recommended_indices = [i for i in top_indices if i not in user_liked_images]

# #         # Fetch LastRecommendedIndex
# #         try:
# #             last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# #         except:
# #             last_recommended_index = 0

# #         # Define batch size
# #         batch_size = 40  # Fetch top 40

# #         # Select the next batch
# #         indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# #         # Update LastRecommendedIndex
# #         new_last_recommended_index = last_recommended_index + batch_size
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# #         save_user_embeddings(user_embeddings)

# #     recommendations = []

# #     for idx in indices:
# #         try:
# #             artwork = train_data[int(idx)]  # Convert to int
# #         except IndexError:
# #             print(f"Index {idx} is out of bounds for the dataset.")
# #             continue
# #         except TypeError as te:
# #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'index': int(idx),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #             'image': img_base64
# #         })

# #     return jsonify({'images': recommendations}), 200

# # @app.route('/chat', methods=['POST'])
# # @token_required
# # def chat(current_user_email):
# #     """
# #     Handle chat requests with text and optional image.
# #     Processes the inputs and returns a response.
# #     """
# #     text = request.form.get('text', '').strip()
# #     image_file = request.files.get('image', None)

# #     image_data = None
# #     if image_file:
# #         try:
# #             image_bytes = image_file.read()
# #             image = Image.open(io.BytesIO(image_bytes))
# #             image = image.convert('RGB')
# #             image_data = image
# #         except Exception as e:
# #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# #     try:
# #         result = predict(text, image_data)
# #         return jsonify(result), 200
# #     except Exception as e:
# #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # def predict(text, image_data=None):
# #     """
# #     Process the input text and image, generate recommendations,
# #     and return them with explanations and metadata.
# #     """
# #     if not all([
# #         all_embeddings is not None, 
# #         train_data is not None, 
# #         clip_model is not None, 
# #         clip_processor is not None, 
# #         blip_model is not None, 
# #         blip_processor is not None
# #     ]):
# #         return {'message': 'Server not fully initialized. Please check the logs.'}

# #     input_image = image_data
# #     user_text = text

# #     if input_image:
# #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# #         print(f"Generated image caption: {image_caption}")
# #     else:
# #         image_caption = ""

# #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# #     print(f"Context-aware text: {context_aware_text}")

# #     if input_image:
# #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# #     else:
# #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# #     inputs = {key: value.to(device) for key, value in inputs.items()}
# #     print("Preprocessed inputs for CLIP.")

# #     with torch.no_grad():
# #         if input_image:
# #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# #             image_features_np = image_features.cpu().detach().numpy()
# #         else:
# #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# #         text_features_np = text_features.cpu().detach().numpy()
# #     print("Generated and normalized image and text features using CLIP.")

# #     weight_img = 0.1
# #     weight_text = 0.9

# #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# #     if norm != 0:
# #         final_embedding = final_embedding / norm
# #     else:
# #         final_embedding = final_embedding
# #     print("Computed final combined embedding.")

# #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# #     embedding_dim = all_embeddings.shape[1]
# #     if final_embedding.shape[1] != embedding_dim:
# #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# #         if final_embedding.shape[1] > embedding_dim:
# #             final_embedding = final_embedding[:, :embedding_dim]
# #             print("Trimmed final_embedding.")
# #         else:
# #             padding_size = embedding_dim - final_embedding.shape[1]
# #             padding = np.zeros((final_embedding.shape[0], padding_size))
# #             final_embedding = np.hstack((final_embedding, padding))
# #             print(f"Padded final_embedding with {padding_size} zeros.")
# #         # Update the embedding in the dataframe
# #         final_embedding_normalized = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# #         # Note: Since this is within predict, not updating user embeddings
# #         # If needed, update here or elsewhere
# #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# #     similarities = cosine_similarity(final_embedding, all_embeddings)
# #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# #     top_n = 10
# #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# #     recommended_artworks = [int(i) for i in top_n_indices]

# #     recommendations = []

# #     for rank, i in enumerate(recommended_artworks, start=1):
# #         try:
# #             artwork = train_data[int(i)]
# #         except IndexError:
# #             print(f"Index {i} is out of bounds for the dataset.")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'rank': rank,
# #             'index': int(i),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             # 'description': curr_metadata['description'],  # Optional: Uncomment if needed
# #             'image': img_base64
# #         })

# #     response_text = "Here are the recommended artworks based on your preferences:"

# #     return {
# #         'response': response_text,
# #         'recommendations': recommendations
# #     }

# # @app.route('/trending', methods=['GET'])
# # @token_required
# # def trending(current_user_email):
# #     """
# #     Retrieves the top 40 trending images based on the number of likes.
# #     Returns the images along with their like counts.
# #     """
# #     try:
# #         image_likes = load_image_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

# #     # Calculate like counts for each image
# #     image_likes['LikeCount'] = image_likes['Users'].apply(len)

# #     # Sort images by LikeCount descendingly
# #     top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

# #     recommendations = []

# #     for _, row in top_images.iterrows():
# #         idx = row['ImageIndex']
# #         like_count = row['LikeCount']

# #         try:
# #             artwork = train_data[int(idx)]
# #         except IndexError:
# #             print(f"Index {idx} is out of bounds for the dataset.")
# #             continue
# #         except TypeError as te:
# #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'index': int(idx),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #             'image': img_base64,
# #             'like_count': like_count
# #         })

# #     return jsonify({'trending_images': recommendations}), 200

# # @app.route('/get_all_liked', methods=['GET'])
# # @token_required
# # def get_all_liked(current_user_email):
# #     """
# #     Retrieves all liked images for the authenticated user.
# #     """
# #     try:
# #         # Load user likes
# #         user_likes = load_user_likes()
# #         liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# #         if not liked_image_indices:
# #             return jsonify({'liked_images': []}), 200

# #         # Fetch image data from train_data
# #         liked_images = []
# #         for idx in liked_image_indices:
# #             try:
# #                 artwork = train_data[int(idx)]
# #             except IndexError:
# #                 print(f"Index {idx} is out of bounds for the dataset.")
# #                 continue
# #             except TypeError as te:
# #                 print(f"TypeError accessing train_data with idx={idx}: {te}")
# #                 continue

# #             curr_metadata = {
# #                 "artist": artwork.get('artist', 'Unknown Artist'),
# #                 "style": artwork.get('style', 'Unknown Style'),
# #                 "genre": artwork.get('genre', 'Unknown Genre'),
# #                 "description": artwork.get('description', 'No Description Available')
# #             }

# #             image_data_or_url = artwork.get('image', None)

# #             if isinstance(image_data_or_url, str):
# #                 try:
# #                     response = requests.get(image_data_or_url)
# #                     if response.status_code == 200:
# #                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                     else:
# #                         artwork_image = None
# #                 except Exception as e:
# #                     print(f"Error fetching image from {image_data_or_url}: {e}")
# #                     artwork_image = None
# #             elif isinstance(image_data_or_url, Image.Image):
# #                 artwork_image = image_data_or_url
# #             else:
# #                 artwork_image = None

# #             if artwork_image:
# #                 try:
# #                     img_base64 = encode_image_to_base64(artwork_image)
# #                 except Exception as e:
# #                     print(f"Error encoding image to base64: {e}")
# #                     img_base64 = None
# #             else:
# #                 img_base64 = None

# #             # Fetch the timestamp of the like
# #             like_timestamp = user_likes[
# #                 (user_likes['UserEmail'] == current_user_email) &
# #                 (user_likes['ImageIndex'] == idx)
# #             ]['LikedAt'].iloc[0]
# #             like_timestamp_iso = like_timestamp.isoformat()

# #             liked_images.append({
# #                 'index': int(idx),  # Convert to int
# #                 'artist': curr_metadata['artist'],
# #                 'style': curr_metadata['style'],
# #                 'genre': curr_metadata['genre'],
# #                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #                 'image': img_base64,
# #                 'timestamp': like_timestamp_iso  # ISO formatted timestamp
# #             })

# #         return jsonify({'liked_images': liked_images}), 200

# #     except Exception as e:
# #         return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500

# # # --- Additional Initialization for image_likes.xlsx ---
# # # Initialize image_likes.xlsx entries for all images if not already present
# # try:
# #     if all_embeddings is not None and train_data is not None:
# #         image_likes = load_image_likes()
# #         existing_indices = set(image_likes['ImageIndex'].tolist())
# #         all_indices = set(range(all_embeddings.shape[0]))
# #         missing_indices = all_indices - existing_indices

# #         if missing_indices:
# #             new_entries = pd.DataFrame({
# #                 'ImageIndex': list(missing_indices),
# #                 'Users': [json.dumps([]) for _ in range(len(missing_indices))]  # Initialize with empty lists
# #             })
# #             image_likes = pd.concat([image_likes, new_entries], ignore_index=True)
# #             save_image_likes(image_likes)
# #             print(f"Initialized likes for {len(missing_indices)} images.")
# #         else:
# #             print("All images already have like entries.")
# #     else:
# #         print("Embeddings or training data not available. Skipping image likes initialization.")
# # except Exception as e:
# #     print(f"Error initializing image likes: {e}")

# # if __name__ == '__main__':
# #     app.run(debug=True)


# # # backend/app.py

# # from flask import Flask, request, jsonify
# # import pandas as pd
# # import os
# # from werkzeug.security import generate_password_hash, check_password_hash
# # import jwt
# # import datetime
# # from functools import wraps
# # from flask_cors import CORS
# # from dotenv import load_dotenv
# # import io
# # from PIL import Image
# # import numpy as np
# # import torch
# # from transformers import CLIPProcessor, CLIPModel
# # from transformers import BlipProcessor, BlipForConditionalGeneration
# # from datasets import load_dataset
# # from sklearn.metrics.pairwise import cosine_similarity
# # import requests
# # import base64
# # import json
# # from filelock import FileLock, Timeout
# # from scipy.sparse import csr_matrix

# # # Optimization: Use caching to minimize Excel read/write operations
# # from threading import Lock

# # def display_image(image_data):
# #     # Function to display images (not used in backend)
# #     pass

# # def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
# #     inputs = blip_processor(images=image, return_tensors="pt").to(device)
# #     with torch.no_grad():
# #         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
# #     caption = blip_processor.decode(out[0], skip_special_tokens=True)
# #     return caption

# # def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
# #     margin = 0.05
# #     if sim_image > sim_text + margin:
# #         reason = "the style and composition of the input image."
# #     elif sim_text > sim_image + margin:
# #         reason = "your textual preferences for nature and the specified colors."
# #     else:
# #         reason = "a balanced combination of both your image and textual preferences."

# #     explanation = (
# #         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
# #         f"is recommended {reason} "
# #         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
# #     )
# #     return explanation

# # def encode_image_to_base64(image):
# #     buffered = io.BytesIO()
# #     image.save(buffered, format="JPEG")
# #     img_bytes = buffered.getvalue()
# #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# #     return img_base64

# # def decode_embedding(embedding_str):
# #     return np.array(json.loads(embedding_str))

# # def encode_embedding(embedding_array):
# #     return json.dumps(embedding_array.tolist())

# # def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
# #     """
# #     Combines the current embedding with the previous one using a weighted average.
# #     """
# #     if previous_embedding is None:
# #         return current_embedding
# #     return weight * current_embedding + (1 - weight) * previous_embedding

# # def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
# #     """
# #     Recommends the top-k similar artworks based on cosine similarity.
# #     """
# #     similarities = cosine_similarity([combined_embedding], all_embeddings)
# #     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
# #     return top_k_indices

# # # Determine the device to use for computation
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(f"Using device: {device}")

# # # Load combined embeddings
# # try:
# #     all_embeddings = np.load('combined_embeddings.npy')
# #     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# # except FileNotFoundError:
# #     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
# #     all_embeddings = None

# # if all_embeddings is not None:
# #     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
# #     print("Normalized all_embeddings for cosine similarity.")
# # else:
# #     print("Skipping normalization due to missing embeddings.")

# # # Load the dataset (e.g., WikiArt for training data)
# # try:
# #     ds = load_dataset("Artificio/WikiArt")
# #     train_data = ds['train']
# #     print("Loaded WikiArt dataset successfully.")
# # except Exception as e:
# #     print(f"Error loading dataset: {e}")
# #     train_data = None

# # # Load CLIP model and processor
# # try:
# #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# #     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# #     clip_model.to(device)
# #     print("Loaded CLIP model and processor successfully.")
# # except Exception as e:
# #     print(f"Error loading CLIP model: {e}")
# #     clip_model = None
# #     clip_processor = None

# # # Load BLIP model and processor for image captioning
# # try:
# #     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# #     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# #     blip_model.to(device)
# #     print("Loaded BLIP model and processor successfully.")
# # except Exception as e:
# #     print(f"Error loading BLIP model: {e}")
# #     blip_model = None
# #     blip_processor = None

# # # Load environment variables from .env file
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)

# # # Retrieve the secret key from environment variables
# # app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # # Ensure that the secret key is set
# # if not app.config['SECRET_KEY']:
# #     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # # Define Excel database files
# # DATABASE_USERS = 'users.xlsx'
# # DATABASE_LIKES = 'user_likes.xlsx'
# # DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'
# # DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# # # Initialize Excel files if they don't exist
# # for db_file, columns in [
# #     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
# #     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
# #     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate']),
# #     (DATABASE_IMAGE_LIKES, ['ImageIndex', 'Users'])  # 'Users' will store JSON strings
# # ]:
# #     if not os.path.exists(db_file):
# #         try:
# #             if db_file == DATABASE_IMAGE_LIKES:
# #                 df = pd.DataFrame(columns=columns)
# #             else:
# #                 df = pd.DataFrame(columns=columns)
# #             df.to_excel(db_file, index=False, engine='openpyxl')
# #             print(f"Created {db_file} with columns: {columns}")
# #         except Exception as e:
# #             print(f"Error creating {db_file}: {e}")

# # # Optimization: Use in-memory caching for Excel data
# # # This cache will store the dataframes and a lock for thread safety
# # data_cache = {
# #     'users': {'data': None, 'lock': Lock()},
# #     'user_likes': {'data': None, 'lock': Lock()},
# #     'user_embeddings': {'data': None, 'lock': Lock()},
# #     'image_likes': {'data': None, 'lock': Lock()}
# # }

# # def load_users():
# #     with data_cache['users']['lock']:
# #         if data_cache['users']['data'] is None:
# #             try:
# #                 data_cache['users']['data'] = pd.read_excel(DATABASE_USERS, engine='openpyxl')
# #             except Exception as e:
# #                 print(f"Error loading {DATABASE_USERS}: {e}")
# #                 data_cache['users']['data'] = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
# #         return data_cache['users']['data']

# # def save_users(df):
# #     with data_cache['users']['lock']:
# #         df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
# #         data_cache['users']['data'] = df

# # def load_user_likes():
# #     with data_cache['user_likes']['lock']:
# #         if data_cache['user_likes']['data'] is None:
# #             try:
# #                 data_cache['user_likes']['data'] = pd.read_excel(DATABASE_LIKES, engine='openpyxl')
# #             except Exception as e:
# #                 print(f"Error loading {DATABASE_LIKES}: {e}")
# #                 data_cache['user_likes']['data'] = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
# #         return data_cache['user_likes']['data']

# # def save_user_likes(df):
# #     with data_cache['user_likes']['lock']:
# #         df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
# #         data_cache['user_likes']['data'] = df

# # def load_user_embeddings():
# #     with data_cache['user_embeddings']['lock']:
# #         if data_cache['user_embeddings']['data'] is None:
# #             try:
# #                 data_cache['user_embeddings']['data'] = pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
# #             except Exception as e:
# #                 print(f"Error loading {DATABASE_EMBEDDINGS}: {e}")
# #                 data_cache['user_embeddings']['data'] = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate'])
# #         return data_cache['user_embeddings']['data']

# # def save_user_embeddings(df):
# #     with data_cache['user_embeddings']['lock']:
# #         df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
# #         data_cache['user_embeddings']['data'] = df

# # def load_image_likes():
# #     with data_cache['image_likes']['lock']:
# #         if data_cache['image_likes']['data'] is None:
# #             try:
# #                 df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
# #                 # Ensure 'Users' column is parsed from JSON strings to lists
# #                 def parse_users(x):
# #                     if isinstance(x, str):
# #                         try:
# #                             return json.loads(x)
# #                         except json.JSONDecodeError:
# #                             return []
# #                     elif isinstance(x, list):
# #                         return x
# #                     else:
# #                         return []

# #                 df['Users'] = df['Users'].apply(parse_users)
# #                 data_cache['image_likes']['data'] = df
# #             except Exception as e:
# #                 print(f"Error loading {DATABASE_IMAGE_LIKES}: {e}")
# #                 data_cache['image_likes']['data'] = pd.DataFrame(columns=['ImageIndex', 'Users'])
# #         return data_cache['image_likes']['data']

# # def save_image_likes(df):
# #     with data_cache['image_likes']['lock']:
# #         # Convert 'Users' lists to JSON strings before saving
# #         df_copy = df.copy()
# #         df_copy['Users'] = df_copy['Users'].apply(lambda x: json.dumps(x))
# #         df_copy.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
# #         data_cache['image_likes']['data'] = df

# # def token_required(f):
# #     @wraps(f)
# #     def decorated(*args, **kwargs):
# #         token = None

# #         if 'Authorization' in request.headers:
# #             auth_header = request.headers['Authorization']
# #             try:
# #                 token = auth_header.split(" ")[1]
# #             except IndexError:
# #                 return jsonify({'message': 'Token format invalid!'}), 401

# #         if not token:
# #             return jsonify({'message': 'Token is missing!'}), 401

# #         try:
# #             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
# #             current_user_email = data['email']
# #         except jwt.ExpiredSignatureError:
# #             return jsonify({'message': 'Token has expired!'}), 401
# #         except jwt.InvalidTokenError:
# #             return jsonify({'message': 'Invalid token!'}), 401

# #         try:
# #             users = load_users()
# #         except Exception as e:
# #             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #         user = users[users['Email'] == current_user_email]
# #         if user.empty:
# #             return jsonify({'message': 'User not found!'}), 401

# #         return f(current_user_email, *args, **kwargs)

# #     return decorated

# # @app.route('/signup', methods=['POST'])
# # def signup():
# #     data = request.get_json()
# #     full_name = data.get('full_name')
# #     email = data.get('email')
# #     password = data.get('password')

# #     if not all([full_name, email, password]):
# #         return jsonify({'message': 'Full name, email, and password are required.'}), 400

# #     try:
# #         users = load_users()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #     if email in users['Email'].values:
# #         return jsonify({'message': 'Email already exists.'}), 400

# #     hashed_password = generate_password_hash(password)

# #     new_user = pd.DataFrame({
# #         'FullName': [full_name],
# #         'Email': [email],
# #         'Password': [hashed_password]
# #     })

# #     try:
# #         users = pd.concat([users, new_user], ignore_index=True)
# #         save_users(users)
# #     except Exception as e:
# #         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

# #     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         if email not in user_embeddings['UserEmail'].values:
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             print(f"Initialized zero embedding for user {email}.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

# #     return jsonify({'message': 'User registered successfully.'}), 201

# # @app.route('/login', methods=['POST'])
# # def login():
# #     data = request.get_json()
# #     email = data.get('email')
# #     password = data.get('password')

# #     if not all([email, password]):
# #         return jsonify({'message': 'Email and password are required.'}), 400

# #     try:
# #         users = load_users()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

# #     user = users[users['Email'] == email]

# #     if user.empty:
# #         return jsonify({'message': 'Invalid email or password.'}), 401

# #     stored_password = user.iloc[0]['Password']
# #     full_name = user.iloc[0]['FullName']

# #     if not check_password_hash(stored_password, password):
# #         return jsonify({'message': 'Invalid email or password.'}), 401

# #     try:
# #         token = jwt.encode({
# #             'email': email,
# #             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
# #         }, app.config['SECRET_KEY'], algorithm="HS256")
# #     except Exception as e:
# #         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

# #     # Ensure user has an embedding; initialize if not
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         if email not in user_embeddings['UserEmail'].values:
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             print(f"Initialized zero embedding for user {email} on login.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

# #     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# # @app.route('/protected', methods=['GET'])
# # @token_required
# # def protected_route(current_user_email):
# #     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # # --- Modified /like-image Endpoint ---
# # @app.route('/like-image', methods=['POST'])
# # @token_required
# # def like_image(current_user_email):
# #     """
# #     Records a user's like for an image and updates embeddings immediately.
# #     Also updates the image_likes.xlsx file to track which users liked which images.
# #     Implements a request queue if the lock is not acquired.
# #     """
# #     data = request.get_json()
# #     image_index = data.get('image_index')

# #     if image_index is None:
# #         return jsonify({'message': 'Image index is required.'}), 400

# #     # Ensure image_index is int
# #     try:
# #         image_index = int(image_index)
# #     except ValueError:
# #         return jsonify({'message': 'Image index must be an integer.'}), 400

# #     # Record the like
# #     try:
# #         user_likes = load_user_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# #     # Ensure image_index is within range
# #     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
# #         return jsonify({'message': 'Invalid image index.'}), 400

# #     new_like = pd.DataFrame({
# #         'UserEmail': [current_user_email],
# #         'ImageIndex': [image_index],
# #         'LikedAt': [datetime.datetime.utcnow()]
# #     })

# #     # Optimization: Append the new like to the in-memory dataframe and schedule a periodic save
# #     with data_cache['user_likes']['lock']:
# #         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
# #         data_cache['user_likes']['data'] = user_likes
# #         # No immediate save to Excel to reduce I/O operations

# #     # --- Update image_likes.xlsx ---
# #     try:
# #         image_likes = load_image_likes()
# #         with data_cache['image_likes']['lock']:
# #             if image_index in image_likes['ImageIndex'].values:
# #                 # Get the current list of users who liked this image
# #                 users = image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'].iloc[0]
# #                 if not isinstance(users, list):
# #                     users = []
# #                 if current_user_email not in users:
# #                     users.append(current_user_email)
# #                     image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'] = [users]
# #                     data_cache['image_likes']['data'] = image_likes
# #                     # No immediate save to Excel to reduce I/O operations
# #                     print(f"Added user {current_user_email} to ImageIndex {image_index} likes.")
# #                 else:
# #                     print(f"User {current_user_email} already liked ImageIndex {image_index}.")
# #             else:
# #                 # If the image index is not present, add it
# #                 new_entry = pd.DataFrame({'ImageIndex': [image_index], 'Users': [[current_user_email]]})
# #                 image_likes = pd.concat([image_likes, new_entry], ignore_index=True)
# #                 data_cache['image_likes']['data'] = image_likes
# #                 # No immediate save to Excel to reduce I/O operations
# #                 print(f"Initialized likes for ImageIndex {image_index} with user {current_user_email}.")
# #     except Exception as e:
# #         return jsonify({'message': f'Error updating image likes: {str(e)}'}), 500

# #     # --- Update user embedding immediately after each like ---
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# #         if user_embedding_row.empty:
# #             # Initialize embedding with zeros if not found
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_user_embedding = pd.DataFrame({
# #                 'UserEmail': [current_user_email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
# #             data_cache['user_embeddings']['data'] = user_embeddings
# #             user_embedding = zero_embedding
# #             print(f"Initialized zero embedding for user {current_user_email} during like update.")
# #         else:
# #             # Decode the existing embedding
# #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])

# #         # Fetch the liked image embedding
# #         if all_embeddings is not None:
# #             liked_embedding = all_embeddings[image_index]
# #             if np.linalg.norm(liked_embedding) != 0:
# #                 liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
# #             else:
# #                 liked_embedding = liked_embedding
# #         else:
# #             # If embeddings are not available, use zero embedding
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             liked_embedding = np.zeros(embedding_dim)

# #         # Combine with previous embedding using the specified weight
# #         weight = 0.2  # Define the weight as needed
# #         combined_embedding = combine_embeddings_for_recommendation(
# #             current_embedding=liked_embedding,
# #             previous_embedding=user_embedding,
# #             weight=weight
# #         )
# #         norm = np.linalg.norm(combined_embedding)
# #         if norm != 0:
# #             combined_embedding = combined_embedding / norm
# #         else:
# #             combined_embedding = combined_embedding

# #         # Update the embedding in the dataframe
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
# #         # Reset LastRecommendedIndex since embedding has been updated
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = 0
# #         # Update LastEmbeddingUpdate timestamp
# #         user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()

# #         data_cache['user_embeddings']['data'] = user_embeddings
# #         # No immediate save to Excel to reduce I/O operations
# #         print(f"Updated embedding for user {current_user_email} after like.")

# #     except Exception as e:
# #         return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

# #     return jsonify({'message': 'Image liked successfully.'}), 200

# # @app.route('/recommend-images', methods=['GET'])
# # @token_required
# # def recommend_images(current_user_email):
# #     """
# #     Provides personalized recommendations based on user embeddings.
# #     """
# #     # Get the collaborative_filtering parameter from query parameters
# #     use_collaborative = request.args.get('collaborative_filtering', 'false').lower() == 'true'

# #     if use_collaborative:
# #         # Use the hybrid recommendation system
# #         return hybrid_recommend(current_user_email)
# #     else:
# #         # Use the content-based recommendation system
# #         try:
# #             user_embeddings = load_user_embeddings()
# #             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# #             if user_embedding_row.empty:
# #                 # Initialize embedding with zeros if not found
# #                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #                 zero_embedding = np.zeros(embedding_dim)
# #                 zero_embedding_encoded = encode_embedding(zero_embedding)
# #                 new_embedding = pd.DataFrame({
# #                     'UserEmail': [current_user_email],
# #                     'Embedding': [zero_embedding_encoded],
# #                     'LastRecommendedIndex': [0],
# #                     'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #                 })
# #                 user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #                 save_user_embeddings(user_embeddings)
# #                 user_embedding = zero_embedding.reshape(1, -1)
# #                 print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
# #             else:
# #                 # Decode the existing embedding
# #                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# #                 last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
# #                 last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# #         except Exception as e:
# #             return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# #         # Check if user has liked any images
# #         try:
# #             user_likes = load_user_likes()
# #         except Exception as e:
# #             return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# #         user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# #         if not user_liked_images:
# #             # User hasn't liked any images yet, return random 40 images
# #             if train_data is not None:
# #                 num_images = len(train_data)
# #                 sample_size = 40 if num_images >= 40 else num_images
# #                 indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
# #             else:
# #                 return jsonify({'message': 'No images available.'}), 500
# #         else:
# #             if all_embeddings is None:
# #                 return jsonify({'message': 'Embeddings not available.'}), 500

# #             # Ensure user_embedding has the correct dimension
# #             embedding_dim = all_embeddings.shape[1]
# #             if user_embedding.shape[1] != embedding_dim:
# #                 if user_embedding.shape[1] > embedding_dim:
# #                     user_embedding = user_embedding[:, :embedding_dim]
# #                     print("Trimmed user_embedding to match embedding_dim.")
# #                 else:
# #                     padding_size = embedding_dim - user_embedding.shape[1]
# #                     padding = np.zeros((user_embedding.shape[0], padding_size))
# #                     user_embedding = np.hstack((user_embedding, padding))
# #                     print(f"Padded user_embedding with {padding_size} zeros.")
# #                 # Update the embedding in the dataframe
# #                 user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
# #                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
# #                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
# #                 save_user_embeddings(user_embeddings)

# #             # Compute similarities
# #             similarities = cosine_similarity(user_embedding, all_embeddings)
# #             top_indices = similarities.argsort()[0][::-1]

# #             # Exclude already liked images
# #             recommended_indices = [i for i in top_indices if i not in user_liked_images]

# #             # Fetch LastRecommendedIndex
# #             try:
# #                 last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
# #             except:
# #                 last_recommended_index = 0

# #             # Define batch size
# #             batch_size = 40  # Fetch top 40

# #             # Select the next batch
# #             indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]

# #             # Update LastRecommendedIndex
# #             new_last_recommended_index = last_recommended_index + batch_size
# #             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
# #             save_user_embeddings(user_embeddings)

# #         recommendations = []

# #         for idx in indices:
# #             try:
# #                 artwork = train_data[int(idx)]  # Convert to int
# #             except IndexError:
# #                 print(f"Index {idx} is out of bounds for the dataset.")
# #                 continue
# #             except TypeError as te:
# #                 print(f"TypeError accessing train_data with idx={idx}: {te}")
# #                 continue

# #             curr_metadata = {
# #                 "artist": artwork.get('artist', 'Unknown Artist'),
# #                 "style": artwork.get('style', 'Unknown Style'),
# #                 "genre": artwork.get('genre', 'Unknown Genre'),
# #                 "description": artwork.get('description', 'No Description Available')
# #             }

# #             image_data_or_url = artwork.get('image', None)

# #             if isinstance(image_data_or_url, str):
# #                 try:
# #                     response = requests.get(image_data_or_url)
# #                     if response.status_code == 200:
# #                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                     else:
# #                         artwork_image = None
# #                 except Exception as e:
# #                     print(f"Error fetching image from {image_data_or_url}: {e}")
# #                     artwork_image = None
# #             elif isinstance(image_data_or_url, Image.Image):
# #                 artwork_image = image_data_or_url
# #             else:
# #                 artwork_image = None

# #             if artwork_image:
# #                 try:
# #                     img_base64 = encode_image_to_base64(artwork_image)
# #                 except Exception as e:
# #                     print(f"Error encoding image to base64: {e}")
# #                     img_base64 = None
# #             else:
# #                 img_base64 = None

# #             recommendations.append({
# #                 'index': int(idx),  # Convert to int
# #                 'artist': curr_metadata['artist'],
# #                 'style': curr_metadata['style'],
# #                 'genre': curr_metadata['genre'],
# #                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #                 'image': img_base64
# #             })

# #         return jsonify({'images': recommendations}), 200

# # # --- Added /hybrid-recommend Endpoint ---
# # def hybrid_recommend(current_user_email):
# #     """
# #     Provides hybrid recommendations combining content-based and collaborative filtering.
# #     """
# #     try:
# #         user_embeddings = load_user_embeddings()
# #         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
# #         if user_embedding_row.empty:
# #             # Initialize embedding with zeros if not found
# #             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
# #             zero_embedding = np.zeros(embedding_dim)
# #             zero_embedding_encoded = encode_embedding(zero_embedding)
# #             new_embedding = pd.DataFrame({
# #                 'UserEmail': [current_user_email],
# #                 'Embedding': [zero_embedding_encoded],
# #                 'LastRecommendedIndex': [0],
# #                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
# #             })
# #             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
# #             save_user_embeddings(user_embeddings)
# #             user_embedding = zero_embedding.reshape(1, -1)
# #             print(f"Initialized zero embedding for user {current_user_email} in hybrid recommend.")
# #         else:
# #             # Decode the existing embedding
# #             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

# #     # Build user-item interaction matrix
# #     try:
# #         user_likes = load_user_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

# #     # Map user emails to user IDs
# #     users = user_likes['UserEmail'].unique().tolist()
# #     user_email_to_id = {email: idx for idx, email in enumerate(users)}
# #     num_users = len(users)
# #     num_items = all_embeddings.shape[0]

# #     # Build user interactions list
# #     user_interactions = []
# #     for _, row in user_likes.iterrows():
# #         user_email = row['UserEmail']
# #         image_index = int(row['ImageIndex'])
# #         user_id = user_email_to_id[user_email]
# #         user_interactions.append([user_id, image_index])

# #     # Build interaction matrix
# #     data = [1] * len(user_interactions)
# #     user_ids = [interaction[0] for interaction in user_interactions]
# #     item_ids = [interaction[1] for interaction in user_interactions]
# #     interaction_matrix = csr_matrix((data, (user_ids, item_ids)), shape=(num_users, num_items))

# #     # Compute user similarity matrix
# #     user_similarity = cosine_similarity(interaction_matrix)

# #     # Get current user's ID
# #     if current_user_email in user_email_to_id:
# #         current_user_id = user_email_to_id[current_user_email]
# #     else:
# #         # If the current user has no interactions yet, return content-based recommendations
# #         current_user_id = None

# #     # Perform hybrid recommendation
# #     if current_user_id is not None:
# #         combined_embedding = user_embedding.flatten()
# #         recommended_indices = hybrid_recommendation(
# #             user_id=current_user_id,
# #             combined_embedding=combined_embedding,
# #             all_embeddings=all_embeddings,
# #             interaction_matrix=interaction_matrix,
# #             user_similarity=user_similarity,
# #             content_weight=0.6,
# #             collaborative_weight=0.4,
# #             top_k=40
# #         )
# #     else:
# #         # Use content-based recommendation if no interactions
# #         similarities = cosine_similarity(user_embedding, all_embeddings)
# #         recommended_indices = similarities.argsort()[0][::-1][:40]

# #     recommendations = []

# #     for idx in recommended_indices:
# #         try:
# #             artwork = train_data[int(idx)]  # Convert to int
# #         except IndexError:
# #             print(f"Index {idx} is out of bounds for the dataset.")
# #             continue
# #         except TypeError as te:
# #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'index': int(idx),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #             'image': img_base64
# #         })

# #     return jsonify({'images': recommendations}), 200

# # def hybrid_recommendation(user_id, combined_embedding, all_embeddings, interaction_matrix, user_similarity, content_weight=0.6, collaborative_weight=0.4, top_k=40):
# #     """
# #     Hybrid recommendation system combining content-based and collaborative filtering.

# #     Args:
# #         user_id: ID of the user to recommend items for.
# #         combined_embedding: Weighted combined embedding for content-based recommendation.
# #         all_embeddings: All artwork embeddings in the dataset.
# #         interaction_matrix: Sparse matrix of user-item interactions.
# #         user_similarity: User similarity matrix.
# #         content_weight: Weight for content-based recommendations.
# #         collaborative_weight: Weight for collaborative recommendations.
# #         top_k: Number of recommendations to generate.

# #     Returns:
# #         List of recommended item indices.
# #     """
# #     # Content-based recommendations
# #     content_similarities = cosine_similarity([combined_embedding], all_embeddings)
# #     content_scores = content_similarities[0]

# #     # Collaborative filtering recommendations
# #     similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude self (at index 0)
# #     target_user_items = set(interaction_matrix[user_id].nonzero()[1])
# #     collaborative_scores = np.zeros(all_embeddings.shape[0])

# #     # Aggregate scores from similar users
# #     for similar_user in similar_users:
# #         similarity_score = user_similarity[user_id, similar_user]
# #         similar_user_items = interaction_matrix[similar_user].nonzero()[1]

# #         for item in similar_user_items:
# #             if item not in target_user_items:  # Exclude already interacted items
# #                 collaborative_scores[item] += similarity_score

# #     # Normalize both scores
# #     content_scores = content_scores / np.max(content_scores) if np.max(content_scores) > 0 else content_scores
# #     collaborative_scores = collaborative_scores / np.max(collaborative_scores) if np.max(collaborative_scores) > 0 else collaborative_scores

# #     # Combine scores using the specified weights
# #     final_scores = content_weight * content_scores + collaborative_weight * collaborative_scores

# #     # Exclude already interacted items
# #     final_scores[list(target_user_items)] = -np.inf

# #     # Get top-k recommendations
# #     recommended_items = np.argsort(-final_scores)[:top_k]
# #     return recommended_items

# # @app.route('/chat', methods=['POST'])
# # @token_required
# # def chat(current_user_email):
# #     """
# #     Handle chat requests with text and optional image.
# #     Processes the inputs and returns a response.
# #     """
# #     text = request.form.get('text', '').strip()
# #     image_file = request.files.get('image', None)

# #     image_data = None
# #     if image_file:
# #         try:
# #             image_bytes = image_file.read()
# #             image = Image.open(io.BytesIO(image_bytes))
# #             image = image.convert('RGB')
# #             image_data = image
# #         except Exception as e:
# #             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

# #     try:
# #         result = predict(text, image_data)
# #         return jsonify(result), 200
# #     except Exception as e:
# #         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# # def predict(text, image_data=None):
# #     """
# #     Process the input text and image, generate recommendations,
# #     and return them with explanations and metadata.
# #     """
# #     if not all([
# #         all_embeddings is not None, 
# #         train_data is not None, 
# #         clip_model is not None, 
# #         clip_processor is not None, 
# #         blip_model is not None, 
# #         blip_processor is not None
# #     ]):
# #         return {'message': 'Server not fully initialized. Please check the logs.'}

# #     input_image = image_data
# #     user_text = text

# #     if input_image:
# #         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
# #         print(f"Generated image caption: {image_caption}")
# #     else:
# #         image_caption = ""

# #     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
# #     print(f"Context-aware text: {context_aware_text}")

# #     if input_image:
# #         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
# #     else:
# #         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
# #     inputs = {key: value.to(device) for key, value in inputs.items()}
# #     print("Preprocessed inputs for CLIP.")

# #     with torch.no_grad():
# #         if input_image:
# #             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
# #             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
# #             image_features_np = image_features.cpu().detach().numpy()
# #         else:
# #             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
# #         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
# #         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
# #         text_features_np = text_features.cpu().detach().numpy()
# #     print("Generated and normalized image and text features using CLIP.")

# #     weight_img = 0.1
# #     weight_text = 0.9

# #     final_embedding = weight_img * image_features_np + weight_text * text_features_np
# #     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
# #     if norm != 0:
# #         final_embedding = final_embedding / norm
# #     else:
# #         final_embedding = final_embedding
# #     print("Computed final combined embedding.")

# #     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
# #     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

# #     embedding_dim = all_embeddings.shape[1]
# #     if final_embedding.shape[1] != embedding_dim:
# #         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
# #         if final_embedding.shape[1] > embedding_dim:
# #             final_embedding = final_embedding[:, :embedding_dim]
# #             print("Trimmed final_embedding.")
# #         else:
# #             padding_size = embedding_dim - final_embedding.shape[1]
# #             padding = np.zeros((final_embedding.shape[0], padding_size))
# #             final_embedding = np.hstack((final_embedding, padding))
# #             print(f"Padded final_embedding with {padding_size} zeros.")
# #         # Update the embedding in the dataframe
# #         final_embedding_normalized = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
# #         # Note: Since this is within predict, not updating user embeddings
# #         # If needed, update here or elsewhere
# #         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

# #     similarities = cosine_similarity(final_embedding, all_embeddings)
# #     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

# #     top_n = 10
# #     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
# #     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

# #     recommended_artworks = [int(i) for i in top_n_indices]

# #     recommendations = []

# #     for rank, i in enumerate(recommended_artworks, start=1):
# #         try:
# #             artwork = train_data[int(i)]
# #         except IndexError:
# #             print(f"Index {i} is out of bounds for the dataset.")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'rank': rank,
# #             'index': int(i),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             # 'description': curr_metadata['description'],  # Optional: Uncomment if needed
# #             'image': img_base64
# #         })

# #     response_text = "Here are the recommended artworks based on your preferences:"

# #     return {
# #         'response': response_text,
# #         'recommendations': recommendations
# #     }

# # @app.route('/trending', methods=['GET'])
# # @token_required
# # def trending(current_user_email):
# #     """
# #     Retrieves the top 40 trending images based on the number of likes.
# #     Returns the images along with their like counts.
# #     """
# #     try:
# #         image_likes = load_image_likes()
# #     except Exception as e:
# #         return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

# #     # Calculate like counts for each image
# #     image_likes['LikeCount'] = image_likes['Users'].apply(len)

# #     # Sort images by LikeCount descendingly
# #     top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

# #     recommendations = []

# #     for _, row in top_images.iterrows():
# #         idx = row['ImageIndex']
# #         like_count = row['LikeCount']

# #         try:
# #             artwork = train_data[int(idx)]
# #         except IndexError:
# #             print(f"Index {idx} is out of bounds for the dataset.")
# #             continue
# #         except TypeError as te:
# #             print(f"TypeError accessing train_data with idx={idx}: {te}")
# #             continue

# #         curr_metadata = {
# #             "artist": artwork.get('artist', 'Unknown Artist'),
# #             "style": artwork.get('style', 'Unknown Style'),
# #             "genre": artwork.get('genre', 'Unknown Genre'),
# #             "description": artwork.get('description', 'No Description Available')
# #         }

# #         image_data_or_url = artwork.get('image', None)

# #         if isinstance(image_data_or_url, str):
# #             try:
# #                 response = requests.get(image_data_or_url)
# #                 if response.status_code == 200:
# #                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                 else:
# #                     artwork_image = None
# #             except Exception as e:
# #                 print(f"Error fetching image from {image_data_or_url}: {e}")
# #                 artwork_image = None
# #         elif isinstance(image_data_or_url, Image.Image):
# #             artwork_image = image_data_or_url
# #         else:
# #             artwork_image = None

# #         if artwork_image:
# #             try:
# #                 img_base64 = encode_image_to_base64(artwork_image)
# #             except Exception as e:
# #                 print(f"Error encoding image to base64: {e}")
# #                 img_base64 = None
# #         else:
# #             img_base64 = None

# #         recommendations.append({
# #             'index': int(idx),  # Convert to int
# #             'artist': curr_metadata['artist'],
# #             'style': curr_metadata['style'],
# #             'genre': curr_metadata['genre'],
# #             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #             'image': img_base64,
# #             'like_count': like_count
# #         })

# #     return jsonify({'trending_images': recommendations}), 200

# # @app.route('/get_all_liked', methods=['GET'])
# # @token_required
# # def get_all_liked(current_user_email):
# #     """
# #     Retrieves all liked images for the authenticated user.
# #     """
# #     try:
# #         # Load user likes
# #         user_likes = load_user_likes()
# #         liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

# #         if not liked_image_indices:
# #             return jsonify({'liked_images': []}), 200

# #         # Fetch image data from train_data
# #         liked_images = []
# #         for idx in liked_image_indices:
# #             try:
# #                 artwork = train_data[int(idx)]
# #             except IndexError:
# #                 print(f"Index {idx} is out of bounds for the dataset.")
# #                 continue
# #             except TypeError as te:
# #                 print(f"TypeError accessing train_data with idx={idx}: {te}")
# #                 continue

# #             curr_metadata = {
# #                 "artist": artwork.get('artist', 'Unknown Artist'),
# #                 "style": artwork.get('style', 'Unknown Style'),
# #                 "genre": artwork.get('genre', 'Unknown Genre'),
# #                 "description": artwork.get('description', 'No Description Available')
# #             }

# #             image_data_or_url = artwork.get('image', None)

# #             if isinstance(image_data_or_url, str):
# #                 try:
# #                     response = requests.get(image_data_or_url)
# #                     if response.status_code == 200:
# #                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
# #                     else:
# #                         artwork_image = None
# #                 except Exception as e:
# #                     print(f"Error fetching image from {image_data_or_url}: {e}")
# #                     artwork_image = None
# #             elif isinstance(image_data_or_url, Image.Image):
# #                 artwork_image = image_data_or_url
# #             else:
# #                 artwork_image = None

# #             if artwork_image:
# #                 try:
# #                     img_base64 = encode_image_to_base64(artwork_image)
# #                 except Exception as e:
# #                     print(f"Error encoding image to base64: {e}")
# #                     img_base64 = None
# #             else:
# #                 img_base64 = None

# #             # Fetch the timestamp of the like
# #             like_timestamp = user_likes[
# #                 (user_likes['UserEmail'] == current_user_email) &
# #                 (user_likes['ImageIndex'] == idx)
# #             ]['LikedAt'].iloc[0]
# #             like_timestamp_iso = like_timestamp.isoformat()

# #             liked_images.append({
# #                 'index': int(idx),  # Convert to int
# #                 'artist': curr_metadata['artist'],
# #                 'style': curr_metadata['style'],
# #                 'genre': curr_metadata['genre'],
# #                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
# #                 'image': img_base64,
# #                 'timestamp': like_timestamp_iso  # ISO formatted timestamp
# #             })

# #         return jsonify({'liked_images': liked_images}), 200

# #     except Exception as e:
# #         return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500

# # # --- Additional Initialization for image_likes.xlsx ---
# # # Initialize image_likes.xlsx entries for all images if not already present
# # try:
# #     if all_embeddings is not None and train_data is not None:
# #         image_likes = load_image_likes()
# #         existing_indices = set(image_likes['ImageIndex'].tolist())
# #         all_indices = set(range(all_embeddings.shape[0]))
# #         missing_indices = all_indices - existing_indices

# #         if missing_indices:
# #             new_entries = pd.DataFrame({
# #                 'ImageIndex': list(missing_indices),
# #                 'Users': [json.dumps([]) for _ in range(len(missing_indices))]  # Initialize with empty lists
# #             })
# #             image_likes = pd.concat([image_likes, new_entries], ignore_index=True)
# #             save_image_likes(image_likes)
# #             print(f"Initialized likes for {len(missing_indices)} images.")
# #         else:
# #             print("All images already have like entries.")
# #     else:
# #         print("Embeddings or training data not available. Skipping image likes initialization.")
# # except Exception as e:
# #     print(f"Error initializing image likes: {e}")

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # backend/app.py

# from flask import Flask, request, jsonify
# import pandas as pd
# import os
# from werkzeug.security import generate_password_hash, check_password_hash
# import jwt
# import datetime
# from functools import wraps
# from flask_cors import CORS
# from dotenv import load_dotenv
# import io
# from PIL import Image
# import numpy as np
# import torch
# from transformers import CLIPProcessor, CLIPModel
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from datasets import load_dataset
# from sklearn.metrics.pairwise import cosine_similarity
# import requests
# import base64
# import json
# from filelock import FileLock, Timeout
# from scipy.sparse import csr_matrix
# import random

# # Optimization: Use caching to minimize Excel read/write operations
# from threading import Lock

# def display_image(image_data):
#     # Function to display images (not used in backend)
#     pass

# def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
#     inputs = blip_processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
#     caption = blip_processor.decode(out[0], skip_special_tokens=True)
#     return caption

# def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
#     margin = 0.05
#     if sim_image > sim_text + margin:
#         reason = "the style and composition of the input image."
#     elif sim_text > sim_image + margin:
#         reason = "your textual preferences for nature and the specified colors."
#     else:
#         reason = "a balanced combination of both your image and textual preferences."

#     explanation = (
#         f"This artwork by {curr_metadata['artist']} in the {curr_metadata['style']} style "
#         f"is recommended {reason} "
#         f"(Image Similarity: {sim_image:.2f}, Text Similarity: {sim_text:.2f})."
#     )
#     return explanation

# def encode_image_to_base64(image):
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     img_bytes = buffered.getvalue()
#     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
#     return img_base64

# def decode_embedding(embedding_str):
#     return np.array(json.loads(embedding_str))

# def encode_embedding(embedding_array):
#     return json.dumps(embedding_array.tolist())

# def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
#     """
#     Combines the current embedding with the previous one using a weighted average.
#     """
#     if previous_embedding is None:
#         return current_embedding
#     return weight * current_embedding + (1 - weight) * previous_embedding

# def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
#     """
#     Recommends the top-k similar artworks based on cosine similarity.
#     """
#     similarities = cosine_similarity([combined_embedding], all_embeddings)
#     top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
#     return top_k_indices

# # Determine the device to use for computation
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load combined embeddings
# try:
#     all_embeddings = np.load('combined_embeddings.npy')
#     print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
# except FileNotFoundError:
#     print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
#     all_embeddings = None

# if all_embeddings is not None:
#     all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
#     print("Normalized all_embeddings for cosine similarity.")
# else:
#     print("Skipping normalization due to missing embeddings.")

# # Load the dataset (e.g., WikiArt for training data)
# try:
#     ds = load_dataset("Artificio/WikiArt")
#     train_data = ds['train']
#     print("Loaded WikiArt dataset successfully.")
# except Exception as e:
#     print(f"Error loading dataset: {e}")
#     train_data = None

# # Load CLIP model and processor
# try:
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     clip_model.to(device)
#     print("Loaded CLIP model and processor successfully.")
# except Exception as e:
#     print(f"Error loading CLIP model: {e}")
#     clip_model = None
#     clip_processor = None

# # Load BLIP model and processor for image captioning
# try:
#     blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     blip_model.to(device)
#     print("Loaded BLIP model and processor successfully.")
# except Exception as e:
#     print(f"Error loading BLIP model: {e}")
#     blip_model = None
#     blip_processor = None

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Retrieve the secret key from environment variables
# app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# # Ensure that the secret key is set
# if not app.config['SECRET_KEY']:
#     raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# # Define Excel database files
# DATABASE_USERS = 'users.xlsx'
# DATABASE_LIKES = 'user_likes.xlsx'
# DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'
# DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# # Initialize Excel files if they don't exist
# for db_file, columns in [
#     (DATABASE_USERS, ['FullName', 'Email', 'Password']),
#     (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
#     (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate', 'LikesSinceLastUpdate', 'AccumulatedEmbedding']),
#     (DATABASE_IMAGE_LIKES, ['ImageIndex', 'Users'])  # 'Users' will store JSON strings
# ]:
#     if not os.path.exists(db_file):
#         try:
#             if db_file == DATABASE_IMAGE_LIKES:
#                 df = pd.DataFrame(columns=columns)
#             else:
#                 df = pd.DataFrame(columns=columns)
#             df.to_excel(db_file, index=False, engine='openpyxl')
#             print(f"Created {db_file} with columns: {columns}")
#         except Exception as e:
#             print(f"Error creating {db_file}: {e}")

# # Optimization: Use in-memory caching for Excel data
# # This cache will store the dataframes and a lock for thread safety
# data_cache = {
#     'users': {'data': None, 'lock': Lock()},
#     'user_likes': {'data': None, 'lock': Lock()},
#     'user_embeddings': {'data': None, 'lock': Lock()},
#     'image_likes': {'data': None, 'lock': Lock()}
# }

# def load_users():
#     with data_cache['users']['lock']:
#         if data_cache['users']['data'] is None:
#             try:
#                 data_cache['users']['data'] = pd.read_excel(DATABASE_USERS, engine='openpyxl')
#             except Exception as e:
#                 print(f"Error loading {DATABASE_USERS}: {e}")
#                 data_cache['users']['data'] = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
#         return data_cache['users']['data']

# def save_users(df):
#     with data_cache['users']['lock']:
#         df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')
#         data_cache['users']['data'] = df

# def load_user_likes():
#     with data_cache['user_likes']['lock']:
#         if data_cache['user_likes']['data'] is None:
#             try:
#                 data_cache['user_likes']['data'] = pd.read_excel(DATABASE_LIKES, engine='openpyxl')
#             except Exception as e:
#                 print(f"Error loading {DATABASE_LIKES}: {e}")
#                 data_cache['user_likes']['data'] = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
#         return data_cache['user_likes']['data']

# def save_user_likes(df):
#     with data_cache['user_likes']['lock']:
#         df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')
#         data_cache['user_likes']['data'] = df

# def load_user_embeddings():
#     with data_cache['user_embeddings']['lock']:
#         if data_cache['user_embeddings']['data'] is None:
#             try:
#                 data_cache['user_embeddings']['data'] = pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
#                 # Ensure 'LikesSinceLastUpdate' and 'AccumulatedEmbedding' columns are present
#                 if 'LikesSinceLastUpdate' not in data_cache['user_embeddings']['data'].columns:
#                     data_cache['user_embeddings']['data']['LikesSinceLastUpdate'] = 0
#                 if 'AccumulatedEmbedding' not in data_cache['user_embeddings']['data'].columns:
#                     embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
#                     zero_embedding_encoded = encode_embedding(np.zeros(embedding_dim))
#                     data_cache['user_embeddings']['data']['AccumulatedEmbedding'] = zero_embedding_encoded
#             except Exception as e:
#                 print(f"Error loading {DATABASE_EMBEDDINGS}: {e}")
#                 data_cache['user_embeddings']['data'] = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate', 'LikesSinceLastUpdate', 'AccumulatedEmbedding'])
#         return data_cache['user_embeddings']['data']

# def save_user_embeddings(df):
#     with data_cache['user_embeddings']['lock']:
#         df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')
#         data_cache['user_embeddings']['data'] = df

# def load_image_likes():
#     with data_cache['image_likes']['lock']:
#         if data_cache['image_likes']['data'] is None:
#             try:
#                 df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
#                 # Ensure 'Users' column is parsed from JSON strings to lists
#                 def parse_users(x):
#                     if isinstance(x, str):
#                         try:
#                             return json.loads(x)
#                         except json.JSONDecodeError:
#                             return []
#                     elif isinstance(x, list):
#                         return x
#                     else:
#                         return []

#                 df['Users'] = df['Users'].apply(parse_users)
#                 data_cache['image_likes']['data'] = df
#             except Exception as e:
#                 print(f"Error loading {DATABASE_IMAGE_LIKES}: {e}")
#                 data_cache['image_likes']['data'] = pd.DataFrame(columns=['ImageIndex', 'Users'])
#         return data_cache['image_likes']['data']

# def save_image_likes(df):
#     with data_cache['image_likes']['lock']:
#         # Convert 'Users' lists to JSON strings before saving
#         df_copy = df.copy()
#         df_copy['Users'] = df_copy['Users'].apply(lambda x: json.dumps(x))
#         df_copy.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')
#         data_cache['image_likes']['data'] = df

# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = None

#         if 'Authorization' in request.headers:
#             auth_header = request.headers['Authorization']
#             try:
#                 token = auth_header.split(" ")[1]
#             except IndexError:
#                 return jsonify({'message': 'Token format invalid!'}), 401

#         if not token:
#             return jsonify({'message': 'Token is missing!'}), 401

#         try:
#             data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
#             current_user_email = data['email']
#         except jwt.ExpiredSignatureError:
#             return jsonify({'message': 'Token has expired!'}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({'message': 'Invalid token!'}), 401

#         try:
#             users = load_users()
#         except Exception as e:
#             return jsonify({'message': f'Error loading users: {str(e)}'}), 500

#         user = users[users['Email'] == current_user_email]
#         if user.empty:
#             return jsonify({'message': 'User not found!'}), 401

#         return f(current_user_email, *args, **kwargs)

#     return decorated

# @app.route('/signup', methods=['POST'])
# def signup():
#     data = request.get_json()
#     full_name = data.get('full_name')
#     email = data.get('email')
#     password = data.get('password')

#     if not all([full_name, email, password]):
#         return jsonify({'message': 'Full name, email, and password are required.'}), 400

#     try:
#         users = load_users()
#     except Exception as e:
#         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

#     if email in users['Email'].values:
#         return jsonify({'message': 'Email already exists.'}), 400

#     hashed_password = generate_password_hash(password)

#     new_user = pd.DataFrame({
#         'FullName': [full_name],
#         'Email': [email],
#         'Password': [hashed_password]
#     })

#     try:
#         users = pd.concat([users, new_user], ignore_index=True)
#         save_users(users)
#     except Exception as e:
#         return jsonify({'message': f'Error saving users: {str(e)}'}), 500

#     # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
#     try:
#         user_embeddings = load_user_embeddings()
#         if email not in user_embeddings['UserEmail'].values:
#             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
#             zero_embedding = np.zeros(embedding_dim)
#             zero_embedding_encoded = encode_embedding(zero_embedding)
#             new_embedding = pd.DataFrame({
#                 'UserEmail': [email],
#                 'Embedding': [zero_embedding_encoded],
#                 'LastRecommendedIndex': [0],
#                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
#                 'LikesSinceLastUpdate': [0],
#                 'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
#             })
#             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
#             save_user_embeddings(user_embeddings)
#             print(f"Initialized zero embedding for user {email}.")
#     except Exception as e:
#         return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

#     return jsonify({'message': 'User registered successfully.'}), 201

# @app.route('/login', methods=['POST'])
# def login():
#     data = request.get_json()
#     email = data.get('email')
#     password = data.get('password')

#     if not all([email, password]):
#         return jsonify({'message': 'Email and password are required.'}), 400

#     try:
#         users = load_users()
#     except Exception as e:
#         return jsonify({'message': f'Error loading users: {str(e)}'}), 500

#     user = users[users['Email'] == email]

#     if user.empty:
#         return jsonify({'message': 'Invalid email or password.'}), 401

#     stored_password = user.iloc[0]['Password']
#     full_name = user.iloc[0]['FullName']

#     if not check_password_hash(stored_password, password):
#         return jsonify({'message': 'Invalid email or password.'}), 401

#     try:
#         token = jwt.encode({
#             'email': email,
#             'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#         }, app.config['SECRET_KEY'], algorithm="HS256")
#     except Exception as e:
#         return jsonify({'message': f'Error generating token: {str(e)}'}), 500

#     # Ensure user has an embedding; initialize if not
#     try:
#         user_embeddings = load_user_embeddings()
#         if email not in user_embeddings['UserEmail'].values:
#             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
#             zero_embedding = np.zeros(embedding_dim)
#             zero_embedding_encoded = encode_embedding(zero_embedding)
#             new_embedding = pd.DataFrame({
#                 'UserEmail': [email],
#                 'Embedding': [zero_embedding_encoded],
#                 'LastRecommendedIndex': [0],
#                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
#                 'LikesSinceLastUpdate': [0],
#                 'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
#             })
#             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
#             save_user_embeddings(user_embeddings)
#             print(f"Initialized zero embedding for user {email} on login.")
#     except Exception as e:
#         return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

#     return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

# @app.route('/protected', methods=['GET'])
# @token_required
# def protected_route(current_user_email):
#     return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

# # --- Modified /like-image Endpoint ---
# @app.route('/like-image', methods=['POST'])
# @token_required
# def like_image(current_user_email):
#     """
#     Records a user's like for an image and updates embeddings after every 5 likes.
#     Also updates the image_likes.xlsx file to track which users liked which images.
#     """
#     data = request.get_json()
#     image_index = data.get('image_index')

#     if image_index is None:
#         return jsonify({'message': 'Image index is required.'}), 400

#     # Ensure image_index is int
#     try:
#         image_index = int(image_index)
#     except ValueError:
#         return jsonify({'message': 'Image index must be an integer.'}), 400

#     # Record the like
#     try:
#         user_likes = load_user_likes()
#     except Exception as e:
#         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

#     # Ensure image_index is within range
#     if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
#         return jsonify({'message': 'Invalid image index.'}), 400

#     new_like = pd.DataFrame({
#         'UserEmail': [current_user_email],
#         'ImageIndex': [image_index],
#         'LikedAt': [datetime.datetime.utcnow()]
#     })

#     # Optimization: Append the new like to the in-memory dataframe and schedule a periodic save
#     with data_cache['user_likes']['lock']:
#         user_likes = pd.concat([user_likes, new_like], ignore_index=True)
#         data_cache['user_likes']['data'] = user_likes
#         # No immediate save to Excel to reduce I/O operations

#     # --- Update image_likes.xlsx ---
#     try:
#         image_likes = load_image_likes()
#         with data_cache['image_likes']['lock']:
#             if image_index in image_likes['ImageIndex'].values:
#                 # Get the current list of users who liked this image
#                 users = image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'].iloc[0]
#                 if not isinstance(users, list):
#                     users = []
#                 if current_user_email not in users:
#                     users.append(current_user_email)
#                     image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'] = [users]
#                     data_cache['image_likes']['data'] = image_likes
#                     # No immediate save to Excel to reduce I/O operations
#                     print(f"Added user {current_user_email} to ImageIndex {image_index} likes.")
#                 else:
#                     print(f"User {current_user_email} already liked ImageIndex {image_index}.")
#             else:
#                 # If the image index is not present, add it
#                 new_entry = pd.DataFrame({'ImageIndex': [image_index], 'Users': [[current_user_email]]})
#                 image_likes = pd.concat([image_likes, new_entry], ignore_index=True)
#                 data_cache['image_likes']['data'] = image_likes
#                 # No immediate save to Excel to reduce I/O operations
#                 print(f"Initialized likes for ImageIndex {image_index} with user {current_user_email}.")
#     except Exception as e:
#         return jsonify({'message': f'Error updating image likes: {str(e)}'}), 500

#     # --- Update user embedding after every 5 likes ---
#     try:
#         user_embeddings = load_user_embeddings()
#         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
#         if user_embedding_row.empty:
#             # Initialize embedding with zeros if not found
#             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
#             zero_embedding = np.zeros(embedding_dim)
#             zero_embedding_encoded = encode_embedding(zero_embedding)
#             new_user_embedding = pd.DataFrame({
#                 'UserEmail': [current_user_email],
#                 'Embedding': [zero_embedding_encoded],
#                 'LastRecommendedIndex': [0],
#                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
#                 'LikesSinceLastUpdate': [1],  # Start with 1 since this is the first like
#                 'AccumulatedEmbedding': [encode_embedding(all_embeddings[image_index])]  # Accumulated embedding is the liked embedding
#             })
#             user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
#             data_cache['user_embeddings']['data'] = user_embeddings
#             print(f"Initialized zero embedding for user {current_user_email} during like update.")
#         else:
#             # Get existing values
#             likes_since_last_update = user_embedding_row.iloc[0].get('LikesSinceLastUpdate', 0)
#             accumulated_embedding_encoded = user_embedding_row.iloc[0].get('AccumulatedEmbedding', None)
#             if accumulated_embedding_encoded is None or pd.isnull(accumulated_embedding_encoded):
#                 accumulated_embedding = np.zeros(all_embeddings.shape[1])
#             else:
#                 accumulated_embedding = decode_embedding(accumulated_embedding_encoded)
#             # Fetch the liked image embedding
#             if all_embeddings is not None:
#                 liked_embedding = all_embeddings[image_index]
#                 if np.linalg.norm(liked_embedding) != 0:
#                     liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
#                 else:
#                     liked_embedding = liked_embedding
#             else:
#                 # If embeddings are not available, use zero embedding
#                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
#                 liked_embedding = np.zeros(embedding_dim)
#             # Add the liked embedding to accumulated_embedding
#             accumulated_embedding += liked_embedding
#             # Increment LikesSinceLastUpdate
#             likes_since_last_update += 1
#             # Update in dataframe
#             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LikesSinceLastUpdate'] = likes_since_last_update
#             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'AccumulatedEmbedding'] = encode_embedding(accumulated_embedding)
#             data_cache['user_embeddings']['data'] = user_embeddings
#             print(f"User {current_user_email} has liked {likes_since_last_update} images since last embedding update.")
#             if likes_since_last_update >= 5:
#                 # Compute average of accumulated embeddings
#                 average_embedding = accumulated_embedding / likes_since_last_update
#                 # Normalize average_embedding
#                 norm = np.linalg.norm(average_embedding)
#                 if norm != 0:
#                     average_embedding = average_embedding / norm
#                 else:
#                     average_embedding = average_embedding
#                 # Decode the existing user embedding
#                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])
#                 # Combine with previous embedding
#                 weight = 0.4  # Adjust weight as needed
#                 combined_embedding = combine_embeddings_for_recommendation(
#                     current_embedding=average_embedding,
#                     previous_embedding=user_embedding,
#                     weight=weight
#                 )
#                 # Normalize combined_embedding
#                 norm = np.linalg.norm(combined_embedding)
#                 if norm != 0:
#                     combined_embedding = combined_embedding / norm
#                 else:
#                     combined_embedding = combined_embedding
#                 # Update 'Embedding' column
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
#                 # Reset 'LikesSinceLastUpdate' and 'AccumulatedEmbedding'
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LikesSinceLastUpdate'] = 0
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'AccumulatedEmbedding'] = encode_embedding(np.zeros(all_embeddings.shape[1]))
#                 # Update 'LastEmbeddingUpdate'
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
#                 data_cache['user_embeddings']['data'] = user_embeddings
#                 # Optionally, save the embeddings now
#                 save_user_embeddings(user_embeddings)
#                 print(f"Updated embedding for user {current_user_email} after 5 likes.")
#     except Exception as e:
#         return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

#     return jsonify({'message': 'Image liked successfully.'}), 200

# # Rest of the code remains the same

# @app.route('/recommend-images', methods=['GET'])
# @token_required
# def recommend_images(current_user_email):
#     """
#     Provides personalized recommendations based on user embeddings.
#     """
#     # Get the collaborative_filtering parameter from query parameters
#     use_collaborative = request.args.get('collaborative_filtering', 'false').lower() == 'true'

#     if use_collaborative:
#         # Use the hybrid recommendation system
#         return hybrid_recommend(current_user_email)
#     else:
#         # Use the content-based recommendation system
#         try:
#             user_embeddings = load_user_embeddings()
#             user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
#             if user_embedding_row.empty:
#                 # Initialize embedding with zeros if not found
#                 embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
#                 zero_embedding = np.zeros(embedding_dim)
#                 zero_embedding_encoded = encode_embedding(zero_embedding)
#                 new_embedding = pd.DataFrame({
#                     'UserEmail': [current_user_email],
#                     'Embedding': [zero_embedding_encoded],
#                     'LastRecommendedIndex': [0],
#                     'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
#                     'LikesSinceLastUpdate': [0],
#                     'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
#                 })
#                 user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
#                 save_user_embeddings(user_embeddings)
#                 user_embedding = zero_embedding.reshape(1, -1)
#                 print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
#             else:
#                 # Decode the existing embedding
#                 user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
#                 last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
#                 last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
#         except Exception as e:
#             return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

#         # Check if user has liked any images
#         try:
#             user_likes = load_user_likes()
#         except Exception as e:
#             return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

#         user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

#         if not user_liked_images:
#             # User hasn't liked any images yet, return random 40 images
#             if train_data is not None:
#                 num_images = len(train_data)
#                 sample_size = 40 if num_images >= 40 else num_images
#                 indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
#             else:
#                 return jsonify({'message': 'No images available.'}), 500
#         else:
#             if all_embeddings is None:
#                 return jsonify({'message': 'Embeddings not available.'}), 500

#             # Ensure user_embedding has the correct dimension
#             embedding_dim = all_embeddings.shape[1]
#             if user_embedding.shape[1] != embedding_dim:
#                 if user_embedding.shape[1] > embedding_dim:
#                     user_embedding = user_embedding[:, :embedding_dim]
#                     print("Trimmed user_embedding to match embedding_dim.")
#                 else:
#                     padding_size = embedding_dim - user_embedding.shape[1]
#                     padding = np.zeros((user_embedding.shape[0], padding_size))
#                     user_embedding = np.hstack((user_embedding, padding))
#                     print(f"Padded user_embedding with {padding_size} zeros.")
#                 # Update the embedding in the dataframe
#                 user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
#                 user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
#                 save_user_embeddings(user_embeddings)

#             # Compute similarities
#             similarities = cosine_similarity(user_embedding, all_embeddings)
#             top_indices = similarities.argsort()[0][::-1]

#             # Exclude already liked images
#             recommended_indices = [i for i in top_indices if i not in user_liked_images]

#             # Fetch LastRecommendedIndex
#             try:
#                 last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
#             except:
#                 last_recommended_index = 0

#             # Define batch size
#             batch_size = 25  # Fetch top 40
#             noise_indices = np.random.choice(len(train_data), size=15, replace=False).tolist()

#             # Select the next batch
#             indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]
#             indices = indices+noise_indices
#             random.shuffle(indices)

#             # Update LastRecommendedIndex
#             new_last_recommended_index = last_recommended_index + batch_size
#             user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
#             save_user_embeddings(user_embeddings)

#         recommendations = []

#         for idx in indices:
#             try:
#                 artwork = train_data[int(idx)]  # Convert to int
#             except IndexError:
#                 print(f"Index {idx} is out of bounds for the dataset.")
#                 continue
#             except TypeError as te:
#                 print(f"TypeError accessing train_data with idx={idx}: {te}")
#                 continue

#             curr_metadata = {
#                 "artist": artwork.get('artist', 'Unknown Artist'),
#                 "style": artwork.get('style', 'Unknown Style'),
#                 "genre": artwork.get('genre', 'Unknown Genre'),
#                 "description": artwork.get('description', 'No Description Available')
#             }

#             image_data_or_url = artwork.get('image', None)

#             if isinstance(image_data_or_url, str):
#                 try:
#                     response = requests.get(image_data_or_url)
#                     if response.status_code == 200:
#                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
#                     else:
#                         artwork_image = None
#                 except Exception as e:
#                     print(f"Error fetching image from {image_data_or_url}: {e}")
#                     artwork_image = None
#             elif isinstance(image_data_or_url, Image.Image):
#                 artwork_image = image_data_or_url
#             else:
#                 artwork_image = None

#             if artwork_image:
#                 try:
#                     img_base64 = encode_image_to_base64(artwork_image)
#                 except Exception as e:
#                     print(f"Error encoding image to base64: {e}")
#                     img_base64 = None
#             else:
#                 img_base64 = None

#             recommendations.append({
#                 'index': int(idx),  # Convert to int
#                 'artist': curr_metadata['artist'],
#                 'style': curr_metadata['style'],
#                 'genre': curr_metadata['genre'],
#                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
#                 'image': img_base64
#             })

#         return jsonify({'images': recommendations}), 200
    
# def hybrid_recommend(current_user_email):
#     """
#     Provides hybrid recommendations combining content-based and collaborative filtering.
#     """
#     try:
#         user_embeddings = load_user_embeddings()
#         user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
#         if user_embedding_row.empty:
#             # Initialize embedding with zeros if not found
#             embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
#             zero_embedding = np.zeros(embedding_dim)
#             zero_embedding_encoded = encode_embedding(zero_embedding)
#             new_embedding = pd.DataFrame({
#                 'UserEmail': [current_user_email],
#                 'Embedding': [zero_embedding_encoded],
#                 'LastRecommendedIndex': [0],
#                 'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
#             })
#             user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
#             save_user_embeddings(user_embeddings)
#             user_embedding = zero_embedding.reshape(1, -1)
#             print(f"Initialized zero embedding for user {current_user_email} in hybrid recommend.")
#         else:
#             # Decode the existing embedding
#             user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
#     except Exception as e:
#         return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

#     # Build user-item interaction matrix
#     try:
#         user_likes = load_user_likes()
#     except Exception as e:
#         return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

#     # Map user emails to user IDs
#     users = user_likes['UserEmail'].unique().tolist()
#     user_email_to_id = {email: idx for idx, email in enumerate(users)}
#     num_users = len(users)
#     num_items = all_embeddings.shape[0]

#     # Build user interactions list
#     user_interactions = []
#     for _, row in user_likes.iterrows():
#         user_email = row['UserEmail']
#         image_index = int(row['ImageIndex'])
#         user_id = user_email_to_id[user_email]
#         user_interactions.append([user_id, image_index])

#     # Build interaction matrix
#     data = [1] * len(user_interactions)
#     user_ids = [interaction[0] for interaction in user_interactions]
#     item_ids = [interaction[1] for interaction in user_interactions]
#     interaction_matrix = csr_matrix((data, (user_ids, item_ids)), shape=(num_users, num_items))

#     # Compute user similarity matrix
#     user_similarity = cosine_similarity(interaction_matrix)

#     # Get current user's ID
#     if current_user_email in user_email_to_id:
#         current_user_id = user_email_to_id[current_user_email]
#     else:
#         # If the current user has no interactions yet, return content-based recommendations
#         current_user_id = None

#     # Perform hybrid recommendation
#     if current_user_id is not None:
#         combined_embedding = user_embedding.flatten()
#         recommended_indices = hybrid_recommendation(
#             user_id=current_user_id,
#             combined_embedding=combined_embedding,
#             all_embeddings=all_embeddings,
#             interaction_matrix=interaction_matrix,
#             user_similarity=user_similarity,
#             content_weight=0.6,
#             collaborative_weight=0.4,
#             top_k=40
#         )
#     else:
#         # Use content-based recommendation if no interactions
#         similarities = cosine_similarity(user_embedding, all_embeddings)
#         recommended_indices = similarities.argsort()[0][::-1][:25]
#         noise_indices = np.random.choice(len(train_data), size=15, replace=False).tolist()
#         recommended_indices = recommended_indices+noise_indices
#         random.shuffle(recommended_indices)

#     recommendations = []

#     for idx in recommended_indices:
#         try:
#             artwork = train_data[int(idx)]  # Convert to int
#         except IndexError:
#             print(f"Index {idx} is out of bounds for the dataset.")
#             continue
#         except TypeError as te:
#             print(f"TypeError accessing train_data with idx={idx}: {te}")
#             continue

#         curr_metadata = {
#             "artist": artwork.get('artist', 'Unknown Artist'),
#             "style": artwork.get('style', 'Unknown Style'),
#             "genre": artwork.get('genre', 'Unknown Genre'),
#             "description": artwork.get('description', 'No Description Available')
#         }

#         image_data_or_url = artwork.get('image', None)

#         if isinstance(image_data_or_url, str):
#             try:
#                 response = requests.get(image_data_or_url)
#                 if response.status_code == 200:
#                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
#                 else:
#                     artwork_image = None
#             except Exception as e:
#                 print(f"Error fetching image from {image_data_or_url}: {e}")
#                 artwork_image = None
#         elif isinstance(image_data_or_url, Image.Image):
#             artwork_image = image_data_or_url
#         else:
#             artwork_image = None

#         if artwork_image:
#             try:
#                 img_base64 = encode_image_to_base64(artwork_image)
#             except Exception as e:
#                 print(f"Error encoding image to base64: {e}")
#                 img_base64 = None
#         else:
#             img_base64 = None

#         recommendations.append({
#             'index': int(idx),  # Convert to int
#             'artist': curr_metadata['artist'],
#             'style': curr_metadata['style'],
#             'genre': curr_metadata['genre'],
#             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
#             'image': img_base64
#         })

#     return jsonify({'images': recommendations}), 200

# def hybrid_recommendation(user_id, combined_embedding, all_embeddings, interaction_matrix, user_similarity, content_weight=0.6, collaborative_weight=0.4, top_k=40):
#     """
#     Hybrid recommendation system combining content-based and collaborative filtering.

#     Args:
#         user_id: ID of the user to recommend items for.
#         combined_embedding: Weighted combined embedding for content-based recommendation.
#         all_embeddings: All artwork embeddings in the dataset.
#         interaction_matrix: Sparse matrix of user-item interactions.
#         user_similarity: User similarity matrix.
#         content_weight: Weight for content-based recommendations.
#         collaborative_weight: Weight for collaborative recommendations.
#         top_k: Number of recommendations to generate.

#     Returns:
#         List of recommended item indices.
#     """
#     # Content-based recommendations
#     content_similarities = cosine_similarity([combined_embedding], all_embeddings)
#     content_scores = content_similarities[0]

#     # Collaborative filtering recommendations
#     similar_users = np.argsort(-user_similarity[user_id])[1:]  # Exclude self (at index 0)
#     target_user_items = set(interaction_matrix[user_id].nonzero()[1])
#     collaborative_scores = np.zeros(all_embeddings.shape[0])

#     # Aggregate scores from similar users
#     for similar_user in similar_users:
#         similarity_score = user_similarity[user_id, similar_user]
#         similar_user_items = interaction_matrix[similar_user].nonzero()[1]

#         for item in similar_user_items:
#             if item not in target_user_items:  # Exclude already interacted items
#                 collaborative_scores[item] += similarity_score

#     # Normalize both scores
#     content_scores = content_scores / np.max(content_scores) if np.max(content_scores) > 0 else content_scores
#     collaborative_scores = collaborative_scores / np.max(collaborative_scores) if np.max(collaborative_scores) > 0 else collaborative_scores

#     # Combine scores using the specified weights
#     final_scores = content_weight * content_scores + collaborative_weight * collaborative_scores

#     # Exclude already interacted items
#     final_scores[list(target_user_items)] = -np.inf

#     # Get top-k recommendations
#     recommended_items = np.argsort(-final_scores)[:top_k]
#     return recommended_items

# @app.route('/chat', methods=['POST'])
# @token_required
# def chat(current_user_email):
#     """
#     Handle chat requests with text and optional image.
#     Processes the inputs and returns a response.
#     """
#     text = request.form.get('text', '').strip()
#     image_file = request.files.get('image', None)

#     image_data = None
#     if image_file:
#         try:
#             image_bytes = image_file.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             image = image.convert('RGB')
#             image_data = image
#         except Exception as e:
#             return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

#     try:
#         result = predict(text, image_data)
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({'message': f'Error processing request: {str(e)}'}), 500

# def predict(text, image_data=None):
#     """
#     Process the input text and image, generate recommendations,
#     and return them with explanations and metadata.
#     """
#     if not all([
#         all_embeddings is not None, 
#         train_data is not None, 
#         clip_model is not None, 
#         clip_processor is not None, 
#         blip_model is not None, 
#         blip_processor is not None
#     ]):
#         return {'message': 'Server not fully initialized. Please check the logs.'}

#     input_image = image_data
#     user_text = text

#     if input_image:
#         image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
#         print(f"Generated image caption: {image_caption}")
#     else:
#         image_caption = ""

#     context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
#     print(f"Context-aware text: {context_aware_text}")

#     if input_image:
#         inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
#     else:
#         inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     print("Preprocessed inputs for CLIP.")

#     with torch.no_grad():
#         if input_image:
#             image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
#             image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#             image_features_np = image_features.cpu().detach().numpy()
#         else:
#             image_features_np = np.zeros((1, clip_model.config.projection_dim))
        
#         text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
#         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#         text_features_np = text_features.cpu().detach().numpy()
#     print("Generated and normalized image and text features using CLIP.")

#     weight_img = 0.1
#     weight_text = 0.9

#     final_embedding = weight_img * image_features_np + weight_text * text_features_np
#     norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
#     if norm != 0:
#         final_embedding = final_embedding / norm
#     else:
#         final_embedding = final_embedding
#     print("Computed final combined embedding.")

#     print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
#     print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

#     embedding_dim = all_embeddings.shape[1]
#     if final_embedding.shape[1] != embedding_dim:
#         print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
#         if final_embedding.shape[1] > embedding_dim:
#             final_embedding = final_embedding[:, :embedding_dim]
#             print("Trimmed final_embedding.")
#         else:
#             padding_size = embedding_dim - final_embedding.shape[1]
#             padding = np.zeros((final_embedding.shape[0], padding_size))
#             final_embedding = np.hstack((final_embedding, padding))
#             print(f"Padded final_embedding with {padding_size} zeros.")
#         # Update the embedding in the dataframe
#         final_embedding_normalized = final_embedding / np.linalg.norm(final_embedding, axis=1, keepdims=True)
#         # Note: Since this is within predict, not updating user embeddings
#         # If needed, update here or elsewhere
#         print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

#     similarities = cosine_similarity(final_embedding, all_embeddings)
#     print("Computed cosine similarities between the final embedding and all dataset embeddings.")

#     top_n = 10
#     top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
#     print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

#     recommended_artworks = [int(i) for i in top_n_indices]

#     recommendations = []

#     for rank, i in enumerate(recommended_artworks, start=1):
#         try:
#             artwork = train_data[int(i)]
#         except IndexError:
#             print(f"Index {i} is out of bounds for the dataset.")
#             continue

#         curr_metadata = {
#             "artist": artwork.get('artist', 'Unknown Artist'),
#             "style": artwork.get('style', 'Unknown Style'),
#             "genre": artwork.get('genre', 'Unknown Genre'),
#             "description": artwork.get('description', 'No Description Available')
#         }

#         image_data_or_url = artwork.get('image', None)

#         if isinstance(image_data_or_url, str):
#             try:
#                 response = requests.get(image_data_or_url)
#                 if response.status_code == 200:
#                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
#                 else:
#                     artwork_image = None
#             except Exception as e:
#                 print(f"Error fetching image from {image_data_or_url}: {e}")
#                 artwork_image = None
#         elif isinstance(image_data_or_url, Image.Image):
#             artwork_image = image_data_or_url
#         else:
#             artwork_image = None

#         if artwork_image:
#             try:
#                 img_base64 = encode_image_to_base64(artwork_image)
#             except Exception as e:
#                 print(f"Error encoding image to base64: {e}")
#                 img_base64 = None
#         else:
#             img_base64 = None

#         recommendations.append({
#             'rank': rank,
#             'index': int(i),  # Convert to int
#             'artist': curr_metadata['artist'],
#             'style': curr_metadata['style'],
#             'genre': curr_metadata['genre'],
#             # 'description': curr_metadata['description'],  # Optional: Uncomment if needed
#             'image': img_base64
#         })

#     response_text = "Here are the recommended artworks based on your preferences:"

#     return {
#         'response': response_text,
#         'recommendations': recommendations
#     }

# @app.route('/trending', methods=['GET'])
# @token_required
# def trending(current_user_email):
#     """
#     Retrieves the top 40 trending images based on the number of likes.
#     Returns the images along with their like counts.
#     """
#     try:
#         image_likes = load_image_likes()
#     except Exception as e:
#         return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

#     # Calculate like counts for each image
#     image_likes['LikeCount'] = image_likes['Users'].apply(len)

#     # Sort images by LikeCount descendingly
#     top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

#     recommendations = []

#     for _, row in top_images.iterrows():
#         idx = row['ImageIndex']
#         like_count = row['LikeCount']

#         try:
#             artwork = train_data[int(idx)]
#         except IndexError:
#             print(f"Index {idx} is out of bounds for the dataset.")
#             continue
#         except TypeError as te:
#             print(f"TypeError accessing train_data with idx={idx}: {te}")
#             continue

#         curr_metadata = {
#             "artist": artwork.get('artist', 'Unknown Artist'),
#             "style": artwork.get('style', 'Unknown Style'),
#             "genre": artwork.get('genre', 'Unknown Genre'),
#             "description": artwork.get('description', 'No Description Available')
#         }

#         image_data_or_url = artwork.get('image', None)

#         if isinstance(image_data_or_url, str):
#             try:
#                 response = requests.get(image_data_or_url)
#                 if response.status_code == 200:
#                     artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
#                 else:
#                     artwork_image = None
#             except Exception as e:
#                 print(f"Error fetching image from {image_data_or_url}: {e}")
#                 artwork_image = None
#         elif isinstance(image_data_or_url, Image.Image):
#             artwork_image = image_data_or_url
#         else:
#             artwork_image = None

#         if artwork_image:
#             try:
#                 img_base64 = encode_image_to_base64(artwork_image)
#             except Exception as e:
#                 print(f"Error encoding image to base64: {e}")
#                 img_base64 = None
#         else:
#             img_base64 = None

#         recommendations.append({
#             'index': int(idx),  # Convert to int
#             'artist': curr_metadata['artist'],
#             'style': curr_metadata['style'],
#             'genre': curr_metadata['genre'],
#             'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
#             'image': img_base64,
#             'like_count': like_count
#         })

#     return jsonify({'trending_images': recommendations}), 200

# @app.route('/get_all_liked', methods=['GET'])
# @token_required
# def get_all_liked(current_user_email):
#     """
#     Retrieves all liked images for the authenticated user.
#     """
#     try:
#         # Load user likes
#         user_likes = load_user_likes()
#         liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

#         if not liked_image_indices:
#             return jsonify({'liked_images': []}), 200

#         # Fetch image data from train_data
#         liked_images = []
#         for idx in liked_image_indices:
#             try:
#                 artwork = train_data[int(idx)]
#             except IndexError:
#                 print(f"Index {idx} is out of bounds for the dataset.")
#                 continue
#             except TypeError as te:
#                 print(f"TypeError accessing train_data with idx={idx}: {te}")
#                 continue

#             curr_metadata = {
#                 "artist": artwork.get('artist', 'Unknown Artist'),
#                 "style": artwork.get('style', 'Unknown Style'),
#                 "genre": artwork.get('genre', 'Unknown Genre'),
#                 "description": artwork.get('description', 'No Description Available')
#             }

#             image_data_or_url = artwork.get('image', None)

#             if isinstance(image_data_or_url, str):
#                 try:
#                     response = requests.get(image_data_or_url)
#                     if response.status_code == 200:
#                         artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
#                     else:
#                         artwork_image = None
#                 except Exception as e:
#                     print(f"Error fetching image from {image_data_or_url}: {e}")
#                     artwork_image = None
#             elif isinstance(image_data_or_url, Image.Image):
#                 artwork_image = image_data_or_url
#             else:
#                 artwork_image = None

#             if artwork_image:
#                 try:
#                     img_base64 = encode_image_to_base64(artwork_image)
#                 except Exception as e:
#                     print(f"Error encoding image to base64: {e}")
#                     img_base64 = None
#             else:
#                 img_base64 = None

#             # Fetch the timestamp of the like
#             like_timestamp = user_likes[
#                 (user_likes['UserEmail'] == current_user_email) &
#                 (user_likes['ImageIndex'] == idx)
#             ]['LikedAt'].iloc[0]
#             like_timestamp_iso = like_timestamp.isoformat()

#             liked_images.append({
#                 'index': int(idx),  # Convert to int
#                 'artist': curr_metadata['artist'],
#                 'style': curr_metadata['style'],
#                 'genre': curr_metadata['genre'],
#                 'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
#                 'image': img_base64,
#                 'timestamp': like_timestamp_iso  # ISO formatted timestamp
#             })

#         return jsonify({'liked_images': liked_images}), 200

#     except Exception as e:
#         return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500

# # --- Additional Initialization for image_likes.xlsx ---
# # Initialize image_likes.xlsx entries for all images if not already present
# try:
#     if all_embeddings is not None and train_data is not None:
#         image_likes = load_image_likes()
#         existing_indices = set(image_likes['ImageIndex'].tolist())
#         all_indices = set(range(all_embeddings.shape[0]))
#         missing_indices = all_indices - existing_indices

#         if missing_indices:
#             new_entries = pd.DataFrame({
#                 'ImageIndex': list(missing_indices),
#                 'Users': [json.dumps([]) for _ in range(len(missing_indices))]  # Initialize with empty lists
#             })
#             image_likes = pd.concat([image_likes, new_entries], ignore_index=True)
#             save_image_likes(image_likes)
#             print(f"Initialized likes for {len(missing_indices)} images.")
#         else:
#             print("All images already have like entries.")
#     else:
#         print("Embeddings or training data not available. Skipping image likes initialization.")
# except Exception as e:
#     print(f"Error initializing image likes: {e}")

# if __name__ == '__main__':
#     app.run(debug=True)


# backend/app.py

from flask import Flask, request, jsonify
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from flask_cors import CORS
from dotenv import load_dotenv
import io
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import requests
import base64
import json
from scipy.sparse import csr_matrix
import random

def display_image(image_data):
    # Function to display images (not used in backend)
    pass

def generate_image_caption(image, blip_model, blip_processor, device, max_new_tokens=50):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_explanation(user_text, curr_metadata, sim_image, sim_text):
    margin = 0.05
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

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def decode_embedding(embedding_str):
    return np.array(json.loads(embedding_str))

def encode_embedding(embedding_array):
    return json.dumps(embedding_array.tolist())

def combine_embeddings_for_recommendation(current_embedding, previous_embedding=None, weight=0.7):
    """
    Combines the current embedding with the previous one using a weighted average.
    """
    if previous_embedding is None:
        return current_embedding
    return weight * current_embedding + (1 - weight) * previous_embedding

def recommend_similar_artworks(combined_embedding, all_embeddings, k=10):
    """
    Recommends the top-k similar artworks based on cosine similarity.
    """
    similarities = cosine_similarity([combined_embedding], all_embeddings)
    top_k_indices = similarities.argsort()[0][::-1][:k]  # Get indices of top-k most similar
    return top_k_indices

# Determine the device to use for computation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load combined embeddings
try:
    all_embeddings = np.load('combined_embeddings.npy')
    print(f"Loaded combined_embeddings.npy with shape: {all_embeddings.shape}")
except FileNotFoundError:
    print("Error: 'combined_embeddings.npy' not found. Please ensure the file exists.")
    all_embeddings = None

if all_embeddings is not None:
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    print("Normalized all_embeddings for cosine similarity.")
else:
    print("Skipping normalization due to missing embeddings.")

# Load the dataset (e.g., WikiArt for training data)
try:
    ds = load_dataset("Artificio/WikiArt")
    train_data = ds['train']
    print("Loaded WikiArt dataset successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    train_data = None

# Load CLIP model and processor
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    print("Loaded CLIP model and processor successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    clip_model = None
    clip_processor = None

# Load BLIP model and processor for image captioning
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to(device)
    print("Loaded BLIP model and processor successfully.")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    blip_model = None
    blip_processor = None

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Retrieve the secret key from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Ensure that the secret key is set
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set for Flask application. Please set the SECRET_KEY environment variable.")

# Define Excel database files
DATABASE_USERS = 'users.xlsx'
DATABASE_LIKES = 'user_likes.xlsx'
DATABASE_EMBEDDINGS = 'user_embeddings.xlsx'
DATABASE_IMAGE_LIKES = 'image_likes.xlsx'

# Initialize Excel files if they don't exist
for db_file, columns in [
    (DATABASE_USERS, ['FullName', 'Email', 'Password']),
    (DATABASE_LIKES, ['UserEmail', 'ImageIndex', 'LikedAt']),
    (DATABASE_EMBEDDINGS, ['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate', 'LikesSinceLastUpdate', 'AccumulatedEmbedding']),
    (DATABASE_IMAGE_LIKES, ['ImageIndex', 'Users'])  # 'Users' will store JSON strings
]:
    if not os.path.exists(db_file):
        try:
            df = pd.DataFrame(columns=columns)
            df.to_excel(db_file, index=False, engine='openpyxl')
            print(f"Created {db_file} with columns: {columns}")
        except Exception as e:
            print(f"Error creating {db_file}: {e}")

def load_users():
    try:
        df = pd.read_excel(DATABASE_USERS, engine='openpyxl')
    except Exception as e:
        print(f"Error loading {DATABASE_USERS}: {e}")
        df = pd.DataFrame(columns=['FullName', 'Email', 'Password'])
    return df

def save_users(df):
    df.to_excel(DATABASE_USERS, index=False, engine='openpyxl')

def load_user_likes():
    try:
        df = pd.read_excel(DATABASE_LIKES, engine='openpyxl')
    except Exception as e:
        print(f"Error loading {DATABASE_LIKES}: {e}")
        df = pd.DataFrame(columns=['UserEmail', 'ImageIndex', 'LikedAt'])
    return df

def save_user_likes(df):
    df.to_excel(DATABASE_LIKES, index=False, engine='openpyxl')

def load_user_embeddings():
    try:
        df = pd.read_excel(DATABASE_EMBEDDINGS, engine='openpyxl')
        # Ensure 'LikesSinceLastUpdate' and 'AccumulatedEmbedding' columns are present
        if 'LikesSinceLastUpdate' not in df.columns:
            df['LikesSinceLastUpdate'] = 0
        if 'AccumulatedEmbedding' not in df.columns:
            embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
            zero_embedding_encoded = encode_embedding(np.zeros(embedding_dim))
            df['AccumulatedEmbedding'] = zero_embedding_encoded
    except Exception as e:
        print(f"Error loading {DATABASE_EMBEDDINGS}: {e}")
        df = pd.DataFrame(columns=['UserEmail', 'Embedding', 'LastRecommendedIndex', 'LastEmbeddingUpdate', 'LikesSinceLastUpdate', 'AccumulatedEmbedding'])
    return df

def save_user_embeddings(df):
    df.to_excel(DATABASE_EMBEDDINGS, index=False, engine='openpyxl')

def load_image_likes():
    try:
        df = pd.read_excel(DATABASE_IMAGE_LIKES, engine='openpyxl')
        # Ensure 'Users' column is parsed from JSON strings to lists
        def parse_users(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return []
            elif isinstance(x, list):
                return x
            else:
                return []
        df['Users'] = df['Users'].apply(parse_users)
    except Exception as e:
        print(f"Error loading {DATABASE_IMAGE_LIKES}: {e}")
        df = pd.DataFrame(columns=['ImageIndex', 'Users'])
    return df

def save_image_likes(df):
    # Convert 'Users' lists to JSON strings before saving
    df_copy = df.copy()
    df_copy['Users'] = df_copy['Users'].apply(lambda x: json.dumps(x))
    df_copy.to_excel(DATABASE_IMAGE_LIKES, index=False, engine='openpyxl')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token format invalid!'}), 401

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_email = data['email']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        try:
            users = load_users()
        except Exception as e:
            return jsonify({'message': f'Error loading users: {str(e)}'}), 500

        user = users[users['Email'] == current_user_email]
        if user.empty:
            return jsonify({'message': 'User not found!'}), 401

        return f(current_user_email, *args, **kwargs)

    return decorated

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')

    if not all([full_name, email, password]):
        return jsonify({'message': 'Full name, email, and password are required.'}), 400

    try:
        users = load_users()
    except Exception as e:
        return jsonify({'message': f'Error loading users: {str(e)}'}), 500

    if email in users['Email'].values:
        return jsonify({'message': 'Email already exists.'}), 400

    hashed_password = generate_password_hash(password)

    new_user = pd.DataFrame({
        'FullName': [full_name],
        'Email': [email],
        'Password': [hashed_password]
    })

    try:
        users = pd.concat([users, new_user], ignore_index=True)
        save_users(users)
    except Exception as e:
        return jsonify({'message': f'Error saving users: {str(e)}'}), 500

    # Initialize user embedding with zeros, LastRecommendedIndex=0, LastEmbeddingUpdate=now
    try:
        user_embeddings = load_user_embeddings()
        if email not in user_embeddings['UserEmail'].values:
            embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
            zero_embedding = np.zeros(embedding_dim)
            zero_embedding_encoded = encode_embedding(zero_embedding)
            new_embedding = pd.DataFrame({
                'UserEmail': [email],
                'Embedding': [zero_embedding_encoded],
                'LastRecommendedIndex': [0],
                'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
                'LikesSinceLastUpdate': [0],
                'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
            })
            user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
            save_user_embeddings(user_embeddings)
            print(f"Initialized zero embedding for user {email}.")
    except Exception as e:
        return jsonify({'message': f'Error initializing user embedding: {str(e)}'}), 500

    return jsonify({'message': 'User registered successfully.'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'message': 'Email and password are required.'}), 400

    try:
        users = load_users()
    except Exception as e:
        return jsonify({'message': f'Error loading users: {str(e)}'}), 500

    user = users[users['Email'] == email]

    if user.empty:
        return jsonify({'message': 'Invalid email or password.'}), 401

    stored_password = user.iloc[0]['Password']
    full_name = user.iloc[0]['FullName']

    if not check_password_hash(stored_password, password):
        return jsonify({'message': 'Invalid email or password.'}), 401

    try:
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm="HS256")
    except Exception as e:
        return jsonify({'message': f'Error generating token: {str(e)}'}), 500

    # Ensure user has an embedding; initialize if not
    try:
        user_embeddings = load_user_embeddings()
        if email not in user_embeddings['UserEmail'].values:
            embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512  # Default to 512 if not available
            zero_embedding = np.zeros(embedding_dim)
            zero_embedding_encoded = encode_embedding(zero_embedding)
            new_embedding = pd.DataFrame({
                'UserEmail': [email],
                'Embedding': [zero_embedding_encoded],
                'LastRecommendedIndex': [0],
                'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
                'LikesSinceLastUpdate': [0],
                'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
            })
            user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
            save_user_embeddings(user_embeddings)
            print(f"Initialized zero embedding for user {email} on login.")
    except Exception as e:
        return jsonify({'message': f'Error initializing user embedding on login: {str(e)}'}), 500

    return jsonify({'message': 'Login successful.', 'token': token, 'full_name': full_name}), 200

@app.route('/protected', methods=['GET'])
@token_required
def protected_route(current_user_email):
    return jsonify({'message': f'Hello, {current_user_email}! This is a protected route.'}), 200

@app.route('/like-image', methods=['POST'])
@token_required
def like_image(current_user_email):
    """
    Records a user's like for an image and updates embeddings after every 5 likes.
    Also updates the image_likes.xlsx file to track which users liked which images.
    """
    data = request.get_json()
    image_index = data.get('image_index')

    if image_index is None:
        return jsonify({'message': 'Image index is required.'}), 400

    # Ensure image_index is int
    try:
        image_index = int(image_index)
    except ValueError:
        return jsonify({'message': 'Image index must be an integer.'}), 400

    # Ensure image_index is within range
    if all_embeddings is not None and not (0 <= image_index < all_embeddings.shape[0]):
        return jsonify({'message': 'Invalid image index.'}), 400

    # Load user_likes
    try:
        user_likes = load_user_likes()
    except Exception as e:
        return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

    # Check if the user has already liked the image
    if not user_likes[
        (user_likes['UserEmail'] == current_user_email) &
        (user_likes['ImageIndex'] == image_index)
    ].empty:
        return jsonify({'message': 'You have already liked this image.'}), 400

    new_like = pd.DataFrame({
        'UserEmail': [current_user_email],
        'ImageIndex': [image_index],
        'LikedAt': [datetime.datetime.utcnow()]
    })

    # Append the new like and save
    user_likes = pd.concat([user_likes, new_like], ignore_index=True)
    save_user_likes(user_likes)
    print(f"Recorded like for user {current_user_email} on image {image_index}.")

    # --- Update image_likes.xlsx ---
    try:
        image_likes = load_image_likes()
        if image_index in image_likes['ImageIndex'].values:
            # Get the current list of users who liked this image
            users = image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'].iloc[0]
            if not isinstance(users, list):
                users = []
            if current_user_email not in users:
                users.append(current_user_email)
                image_likes.loc[image_likes['ImageIndex'] == image_index, 'Users'] = [users]
                save_image_likes(image_likes)
                print(f"Added user {current_user_email} to ImageIndex {image_index} likes.")
            else:
                print(f"User {current_user_email} already liked ImageIndex {image_index}.")
        else:
            # If the image index is not present, add it
            new_entry = pd.DataFrame({'ImageIndex': [image_index], 'Users': [[current_user_email]]})
            image_likes = pd.concat([image_likes, new_entry], ignore_index=True)
            save_image_likes(image_likes)
            print(f"Initialized likes for ImageIndex {image_index} with user {current_user_email}.")
    except Exception as e:
        return jsonify({'message': f'Error updating image likes: {str(e)}'}), 500

    # --- Update user embedding after every 5 likes ---
    try:
        user_embeddings = load_user_embeddings()
        user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
        if user_embedding_row.empty:
            # Initialize embedding with zeros if not found
            embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
            zero_embedding = np.zeros(embedding_dim)
            zero_embedding_encoded = encode_embedding(zero_embedding)
            new_user_embedding = pd.DataFrame({
                'UserEmail': [current_user_email],
                'Embedding': [zero_embedding_encoded],
                'LastRecommendedIndex': [0],
                'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
                'LikesSinceLastUpdate': [1],  # Start with 1 since this is the first like
                'AccumulatedEmbedding': [encode_embedding(all_embeddings[image_index])]  # Accumulated embedding is the liked embedding
            })
            user_embeddings = pd.concat([user_embeddings, new_user_embedding], ignore_index=True)
            save_user_embeddings(user_embeddings)
            print(f"Initialized zero embedding for user {current_user_email} during like update.")
        else:
            # Get existing values
            likes_since_last_update = user_embedding_row.iloc[0].get('LikesSinceLastUpdate', 0)
            accumulated_embedding_encoded = user_embedding_row.iloc[0].get('AccumulatedEmbedding', None)
            if accumulated_embedding_encoded is None or pd.isnull(accumulated_embedding_encoded):
                accumulated_embedding = np.zeros(all_embeddings.shape[1])
            else:
                accumulated_embedding = decode_embedding(accumulated_embedding_encoded)
            # Fetch the liked image embedding
            if all_embeddings is not None:
                liked_embedding = all_embeddings[image_index]
                if np.linalg.norm(liked_embedding) != 0:
                    liked_embedding = liked_embedding / np.linalg.norm(liked_embedding)
                else:
                    liked_embedding = liked_embedding
            else:
                # If embeddings are not available, use zero embedding
                embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
                liked_embedding = np.zeros(embedding_dim)
            # Add the liked embedding to accumulated_embedding
            accumulated_embedding += liked_embedding
            # Increment LikesSinceLastUpdate
            likes_since_last_update += 1
            # Update in dataframe
            user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LikesSinceLastUpdate'] = likes_since_last_update
            user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'AccumulatedEmbedding'] = encode_embedding(accumulated_embedding)
            save_user_embeddings(user_embeddings)
            print(f"User {current_user_email} has liked {likes_since_last_update} images since last embedding update.")
            if likes_since_last_update >= 5:
                # Compute average of accumulated embeddings
                average_embedding = accumulated_embedding / likes_since_last_update
                # Normalize average_embedding
                norm = np.linalg.norm(average_embedding)
                if norm != 0:
                    average_embedding = average_embedding / norm
                else:
                    average_embedding = average_embedding
                # Decode the existing user embedding
                user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding'])
                # Combine with previous embedding
                weight = 0.4  # Adjust weight as needed
                combined_embedding = combine_embeddings_for_recommendation(
                    current_embedding=average_embedding,
                    previous_embedding=user_embedding,
                    weight=weight
                )
                # Normalize combined_embedding
                norm = np.linalg.norm(combined_embedding)
                if norm != 0:
                    combined_embedding = combined_embedding / norm
                else:
                    combined_embedding = combined_embedding
                # Update 'Embedding' column
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(combined_embedding)
                # Reset 'LikesSinceLastUpdate' and 'AccumulatedEmbedding'
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LikesSinceLastUpdate'] = 0
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'AccumulatedEmbedding'] = encode_embedding(np.zeros(all_embeddings.shape[1]))
                # Update 'LastEmbeddingUpdate'
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
                save_user_embeddings(user_embeddings)
                print(f"Updated embedding for user {current_user_email} after 5 likes.")
    except Exception as e:
        return jsonify({'message': f'Error updating user embeddings: {str(e)}'}), 500

    return jsonify({'message': 'Image liked successfully.'}), 200

@app.route('/recommend-images', methods=['GET'])
@token_required
def recommend_images(current_user_email):
    """
    Provides personalized recommendations based on user embeddings.
    """
    # Get the collaborative_filtering parameter from query parameters
    use_collaborative = request.args.get('collaborative_filtering', 'false').lower() == 'true'

    if use_collaborative:
        # Use the hybrid recommendation system
        return hybrid_recommend(current_user_email)
    else:
        # Use the content-based recommendation system
        try:
            user_embeddings = load_user_embeddings()
            user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
            if user_embedding_row.empty:
                # Initialize embedding with zeros if not found
                embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
                zero_embedding = np.zeros(embedding_dim)
                zero_embedding_encoded = encode_embedding(zero_embedding)
                new_embedding = pd.DataFrame({
                    'UserEmail': [current_user_email],
                    'Embedding': [zero_embedding_encoded],
                    'LastRecommendedIndex': [0],
                    'LastEmbeddingUpdate': [datetime.datetime.utcnow()],
                    'LikesSinceLastUpdate': [0],
                    'AccumulatedEmbedding': [encode_embedding(np.zeros(embedding_dim))]
                })
                user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
                save_user_embeddings(user_embeddings)
                user_embedding = zero_embedding.reshape(1, -1)
                print(f"Initialized zero embedding for user {current_user_email} in /recommend-images.")
            else:
                # Decode the existing embedding
                user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
                last_embedding_update = user_embedding_row.iloc[0]['LastEmbeddingUpdate']
                last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
        except Exception as e:
            return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

        # Check if user has liked any images
        try:
            user_likes = load_user_likes()
        except Exception as e:
            return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

        user_liked_images = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

        if not user_liked_images:
            # User hasn't liked any images yet, return random 40 images
            if train_data is not None:
                num_images = len(train_data)
                sample_size = 60 if num_images >= 60 else num_images
                # indices = np.random.choice(num_images, size=sample_size, replace=False).tolist()
                indices = []
                tempsum = 0
                for i in range(sample_size):
                    tempsum+=i
                    indices.append(tempsum)
            else:
                return jsonify({'message': 'No images available.'}), 500
        else:
            if all_embeddings is None:
                return jsonify({'message': 'Embeddings not available.'}), 500

            # Ensure user_embedding has the correct dimension
            embedding_dim = all_embeddings.shape[1]
            if user_embedding.shape[1] != embedding_dim:
                if user_embedding.shape[1] > embedding_dim:
                    user_embedding = user_embedding[:, :embedding_dim]
                    print("Trimmed user_embedding to match embedding_dim.")
                else:
                    padding_size = embedding_dim - user_embedding.shape[1]
                    padding = np.zeros((user_embedding.shape[0], padding_size))
                    user_embedding = np.hstack((user_embedding, padding))
                    print(f"Padded user_embedding with {padding_size} zeros.")
                # Update the embedding in the dataframe
                user_embedding_normalized = user_embedding / np.linalg.norm(user_embedding, axis=1, keepdims=True)
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'Embedding'] = encode_embedding(user_embedding_normalized[0])
                user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastEmbeddingUpdate'] = datetime.datetime.utcnow()
                save_user_embeddings(user_embeddings)

            # Compute similarities
            similarities = cosine_similarity(user_embedding, all_embeddings)
            top_indices = similarities.argsort()[0][::-1]

            # Exclude already liked images
            recommended_indices = [i for i in top_indices if i not in user_liked_images]

            # Fetch LastRecommendedIndex
            try:
                last_recommended_index = user_embedding_row.iloc[0]['LastRecommendedIndex']
            except:
                last_recommended_index = 0

            # Define batch size
            batch_size = 25  # Fetch top 25
            noise_indices = np.random.choice(len(train_data), size=15, replace=False).tolist()

            # Select the next batch
            indices = recommended_indices[last_recommended_index:last_recommended_index + batch_size]
            indices = indices + noise_indices
            random.shuffle(indices)

            # Update LastRecommendedIndex
            new_last_recommended_index = last_recommended_index + batch_size
            user_embeddings.loc[user_embeddings['UserEmail'] == current_user_email, 'LastRecommendedIndex'] = new_last_recommended_index
            save_user_embeddings(user_embeddings)

        recommendations = []

        for idx in indices:
            try:
                artwork = train_data[int(idx)]  # Convert to int
            except IndexError:
                print(f"Index {idx} is out of bounds for the dataset.")
                continue
            except TypeError as te:
                print(f"TypeError accessing train_data with idx={idx}: {te}")
                continue

            curr_metadata = {
                "artist": artwork.get('artist', 'Unknown Artist'),
                "style": artwork.get('style', 'Unknown Style'),
                "genre": artwork.get('genre', 'Unknown Genre'),
                "description": artwork.get('description', 'No Description Available')
            }

            image_data_or_url = artwork.get('image', None)

            if isinstance(image_data_or_url, str):
                try:
                    response = requests.get(image_data_or_url)
                    if response.status_code == 200:
                        artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    else:
                        artwork_image = None
                except Exception as e:
                    print(f"Error fetching image from {image_data_or_url}: {e}")
                    artwork_image = None
            elif isinstance(image_data_or_url, Image.Image):
                artwork_image = image_data_or_url
            else:
                artwork_image = None

            if artwork_image:
                try:
                    img_base64 = encode_image_to_base64(artwork_image)
                except Exception as e:
                    print(f"Error encoding image to base64: {e}")
                    img_base64 = None
            else:
                img_base64 = None

            recommendations.append({
                'index': int(idx),  # Convert to int
                'artist': curr_metadata['artist'],
                'style': curr_metadata['style'],
                'genre': curr_metadata['genre'],
                'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
                'image': img_base64
            })

        return jsonify({'images': recommendations}), 200

def hybrid_recommend(current_user_email):
    """
    Provides hybrid recommendations combining content-based and collaborative filtering.
    """
    try:
        user_embeddings = load_user_embeddings()
        user_embedding_row = user_embeddings[user_embeddings['UserEmail'] == current_user_email]
        if user_embedding_row.empty:
            # Initialize embedding with zeros if not found
            embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 512
            zero_embedding = np.zeros(embedding_dim)
            zero_embedding_encoded = encode_embedding(zero_embedding)
            new_embedding = pd.DataFrame({
                'UserEmail': [current_user_email],
                'Embedding': [zero_embedding_encoded],
                'LastRecommendedIndex': [0],
                'LastEmbeddingUpdate': [datetime.datetime.utcnow()]
            })
            user_embeddings = pd.concat([user_embeddings, new_embedding], ignore_index=True)
            save_user_embeddings(user_embeddings)
            user_embedding = zero_embedding.reshape(1, -1)
            print(f"Initialized zero embedding for user {current_user_email} in hybrid recommend.")
        else:
            # Decode the existing embedding
            user_embedding = decode_embedding(user_embedding_row.iloc[0]['Embedding']).reshape(1, -1)
    except Exception as e:
        return jsonify({'message': f'Error loading user embeddings: {str(e)}'}), 500

    # Build user-item interaction matrix
    try:
        user_likes = load_user_likes()
    except Exception as e:
        return jsonify({'message': f'Error loading user likes: {str(e)}'}), 500

    # Map user emails to user IDs
    users = user_likes['UserEmail'].unique().tolist()
    user_email_to_id = {email: idx for idx, email in enumerate(users)}
    num_users = len(users)
    num_items = all_embeddings.shape[0]

    # Build user interactions list
    user_interactions = []
    for _, row in user_likes.iterrows():
        user_email = row['UserEmail']
        image_index = int(row['ImageIndex'])
        user_id = user_email_to_id[user_email]
        user_interactions.append([user_id, image_index])

    # Build interaction matrix
    data = [1] * len(user_interactions)
    user_ids = [interaction[0] for interaction in user_interactions]
    item_ids = [interaction[1] for interaction in user_interactions]
    interaction_matrix = csr_matrix((data, (user_ids, item_ids)), shape=(num_users, num_items))

    # Compute user similarity matrix
    user_similarity = cosine_similarity(interaction_matrix)

    # Get current user's ID
    if current_user_email in user_email_to_id:
        current_user_id = user_email_to_id[current_user_email]
    else:
        # If the current user has no interactions yet, return content-based recommendations
        current_user_id = None

    # Perform hybrid recommendation
    if current_user_id is not None:
        combined_embedding = user_embedding.flatten()
        recommended_indices = hybrid_recommendation(
            user_id=current_user_id,
            combined_embedding=combined_embedding,
            all_embeddings=all_embeddings,
            interaction_matrix=interaction_matrix,
            user_similarity=user_similarity,
            content_weight=0.4,
            collaborative_weight=0.6,
            top_k=40
        )
    else:
        # Use content-based recommendation if no interactions
        similarities = cosine_similarity(user_embedding, all_embeddings)
        recommended_indices = similarities.argsort()[0][::-1][:25]
        noise_indices = np.random.choice(len(train_data), size=15, replace=False).tolist()
        recommended_indices = recommended_indices + noise_indices
        random.shuffle(recommended_indices)

    recommendations = []

    for idx in recommended_indices:
        try:
            artwork = train_data[int(idx)]  # Convert to int
        except IndexError:
            print(f"Index {idx} is out of bounds for the dataset.")
            continue
        except TypeError as te:
            print(f"TypeError accessing train_data with idx={idx}: {te}")
            continue

        curr_metadata = {
            "artist": artwork.get('artist', 'Unknown Artist'),
            "style": artwork.get('style', 'Unknown Style'),
            "genre": artwork.get('genre', 'Unknown Genre'),
            "description": artwork.get('description', 'No Description Available')
        }

        image_data_or_url = artwork.get('image', None)

        if isinstance(image_data_or_url, str):
            try:
                response = requests.get(image_data_or_url)
                if response.status_code == 200:
                    artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                else:
                    artwork_image = None
            except Exception as e:
                print(f"Error fetching image from {image_data_or_url}: {e}")
                artwork_image = None
        elif isinstance(image_data_or_url, Image.Image):
            artwork_image = image_data_or_url
        else:
            artwork_image = None

        if artwork_image:
            try:
                img_base64 = encode_image_to_base64(artwork_image)
            except Exception as e:
                print(f"Error encoding image to base64: {e}")
                img_base64 = None
        else:
            img_base64 = None

        recommendations.append({
            'index': int(idx),  # Convert to int
            'artist': curr_metadata['artist'],
            'style': curr_metadata['style'],
            'genre': curr_metadata['genre'],
            'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
            'image': img_base64
        })

    return jsonify({'images': recommendations}), 200

def hybrid_recommendation(user_id, combined_embedding, all_embeddings, interaction_matrix, user_similarity, content_weight=0.6, collaborative_weight=0.4, top_k=40):
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

    # Exclude already interacted items
    final_scores[list(target_user_items)] = -np.inf

    # Get top-k recommendations
    recommended_items = np.argsort(-final_scores)[:top_k]
    return recommended_items

@app.route('/chat', methods=['POST'])
@token_required
def chat(current_user_email):
    """
    Handle chat requests with text and optional image.
    Processes the inputs and returns a response.
    """
    text = request.form.get('text', '').strip()
    image_file = request.files.get('image', None)

    image_data = None
    if image_file:
        try:
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            image_data = image
        except Exception as e:
            return jsonify({'message': f'Invalid image file: {str(e)}'}), 400

    try:
        result = predict(text, image_data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'message': f'Error processing request: {str(e)}'}), 500

def predict(text, image_data=None):
    """
    Process the input text and image, generate recommendations,
    and return them with explanations and metadata.
    """
    if not all([
        all_embeddings is not None,
        train_data is not None,
        clip_model is not None,
        clip_processor is not None,
        blip_model is not None,
        blip_processor is not None
    ]):
        return {'message': 'Server not fully initialized. Please check the logs.'}

    input_image = image_data
    user_text = text

    if input_image:
        image_caption = generate_image_caption(input_image, blip_model, blip_processor, device)
        print(f"Generated image caption: {image_caption}")
    else:
        image_caption = ""

    context_aware_text = f"The given image is {image_caption}. {user_text}" if image_caption else user_text
    print(f"Context-aware text: {context_aware_text}")

    if input_image:
        inputs = clip_processor(text=[context_aware_text], images=input_image, return_tensors="pt", padding=True)
    else:
        inputs = clip_processor(text=[context_aware_text], images=None, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print("Preprocessed inputs for CLIP.")

    with torch.no_grad():
        if input_image:
            image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            image_features_np = image_features.cpu().detach().numpy()
        else:
            image_features_np = np.zeros((1, clip_model.config.projection_dim))

        text_features = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features_np = text_features.cpu().detach().numpy()
    print("Generated and normalized image and text features using CLIP.")

    weight_img = 0.1
    weight_text = 0.9

    final_embedding = weight_img * image_features_np + weight_text * text_features_np
    norm = np.linalg.norm(final_embedding, axis=1, keepdims=True)
    if norm != 0:
        final_embedding = final_embedding / norm
    else:
        final_embedding = final_embedding
    print("Computed final combined embedding.")

    print(f"Shape of final_embedding: {final_embedding.shape}")  # Should be (1, embedding_dim)
    print(f"Shape of all_embeddings: {all_embeddings.shape}")    # Should be (num_artworks, embedding_dim)

    embedding_dim = all_embeddings.shape[1]
    if final_embedding.shape[1] != embedding_dim:
        print(f"Adjusting final_embedding from {final_embedding.shape[1]} to {embedding_dim} dimensions.")
        if final_embedding.shape[1] > embedding_dim:
            final_embedding = final_embedding[:, :embedding_dim]
            print("Trimmed final_embedding.")
        else:
            padding_size = embedding_dim - final_embedding.shape[1]
            padding = np.zeros((final_embedding.shape[0], padding_size))
            final_embedding = np.hstack((final_embedding, padding))
            print(f"Padded final_embedding with {padding_size} zeros.")
        print(f"Adjusted final_embedding shape: {final_embedding.shape}")  # Should now be (1, embedding_dim)

    similarities = cosine_similarity(final_embedding, all_embeddings)
    print("Computed cosine similarities between the final embedding and all dataset embeddings.")

    top_n = 10
    top_n_indices = np.argsort(similarities[0])[::-1][:top_n]
    print(f"Top {top_n} recommended artwork indices: {top_n_indices.tolist()}")

    recommended_artworks = [int(i) for i in top_n_indices]

    recommendations = []

    for rank, i in enumerate(recommended_artworks, start=1):
        try:
            artwork = train_data[int(i)]
        except IndexError:
            print(f"Index {i} is out of bounds for the dataset.")
            continue

        curr_metadata = {
            "artist": artwork.get('artist', 'Unknown Artist'),
            "style": artwork.get('style', 'Unknown Style'),
            "genre": artwork.get('genre', 'Unknown Genre'),
            "description": artwork.get('description', 'No Description Available')
        }

        image_data_or_url = artwork.get('image', None)

        if isinstance(image_data_or_url, str):
            try:
                response = requests.get(image_data_or_url)
                if response.status_code == 200:
                    artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                else:
                    artwork_image = None
            except Exception as e:
                print(f"Error fetching image from {image_data_or_url}: {e}")
                artwork_image = None
        elif isinstance(image_data_or_url, Image.Image):
            artwork_image = image_data_or_url
        else:
            artwork_image = None

        if artwork_image:
            try:
                img_base64 = encode_image_to_base64(artwork_image)
            except Exception as e:
                print(f"Error encoding image to base64: {e}")
                img_base64 = None
        else:
            img_base64 = None

        recommendations.append({
            'rank': rank,
            'index': int(i),  # Convert to int
            'artist': curr_metadata['artist'],
            'style': curr_metadata['style'],
            'genre': curr_metadata['genre'],
            'image': img_base64
        })

    response_text = "Here are the recommended artworks based on your preferences:"

    return {
        'response': response_text,
        'recommendations': recommendations
    }

@app.route('/trending', methods=['GET'])
@token_required
def trending(current_user_email):
    """
    Retrieves the top 40 trending images based on the number of likes.
    Returns the images along with their like counts.
    """
    try:
        image_likes = load_image_likes()
    except Exception as e:
        return jsonify({'message': f'Error loading image likes: {str(e)}'}), 500

    # Calculate like counts for each image
    image_likes['LikeCount'] = image_likes['Users'].apply(len)

    # Sort images by LikeCount descendingly
    top_images = image_likes.sort_values(by='LikeCount', ascending=False).head(40)

    recommendations = []

    for _, row in top_images.iterrows():
        idx = row['ImageIndex']
        like_count = row['LikeCount']

        try:
            artwork = train_data[int(idx)]
        except IndexError:
            print(f"Index {idx} is out of bounds for the dataset.")
            continue
        except TypeError as te:
            print(f"TypeError accessing train_data with idx={idx}: {te}")
            continue

        curr_metadata = {
            "artist": artwork.get('artist', 'Unknown Artist'),
            "style": artwork.get('style', 'Unknown Style'),
            "genre": artwork.get('genre', 'Unknown Genre'),
            "description": artwork.get('description', 'No Description Available')
        }

        image_data_or_url = artwork.get('image', None)

        if isinstance(image_data_or_url, str):
            try:
                response = requests.get(image_data_or_url)
                if response.status_code == 200:
                    artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                else:
                    artwork_image = None
            except Exception as e:
                print(f"Error fetching image from {image_data_or_url}: {e}")
                artwork_image = None
        elif isinstance(image_data_or_url, Image.Image):
            artwork_image = image_data_or_url
        else:
            artwork_image = None

        if artwork_image:
            try:
                img_base64 = encode_image_to_base64(artwork_image)
            except Exception as e:
                print(f"Error encoding image to base64: {e}")
                img_base64 = None
        else:
            img_base64 = None

        recommendations.append({
            'index': int(idx),  # Convert to int
            'artist': curr_metadata['artist'],
            'style': curr_metadata['style'],
            'genre': curr_metadata['genre'],
            'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
            'image': img_base64,
            'like_count': like_count
        })

    return jsonify({'trending_images': recommendations}), 200

@app.route('/get_all_liked', methods=['GET'])
@token_required
def get_all_liked(current_user_email):
    """
    Retrieves all liked images for the authenticated user.
    """
    try:
        # Load user likes
        user_likes = load_user_likes()
        liked_image_indices = user_likes[user_likes['UserEmail'] == current_user_email]['ImageIndex'].tolist()

        if not liked_image_indices:
            return jsonify({'liked_images': []}), 200

        # Fetch image data from train_data
        liked_images = []
        for idx in liked_image_indices:
            try:
                artwork = train_data[int(idx)]
            except IndexError:
                print(f"Index {idx} is out of bounds for the dataset.")
                continue
            except TypeError as te:
                print(f"TypeError accessing train_data with idx={idx}: {te}")
                continue

            curr_metadata = {
                "artist": artwork.get('artist', 'Unknown Artist'),
                "style": artwork.get('style', 'Unknown Style'),
                "genre": artwork.get('genre', 'Unknown Genre'),
                "description": artwork.get('description', 'No Description Available')
            }

            image_data_or_url = artwork.get('image', None)

            if isinstance(image_data_or_url, str):
                try:
                    response = requests.get(image_data_or_url)
                    if response.status_code == 200:
                        artwork_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    else:
                        artwork_image = None
                except Exception as e:
                    print(f"Error fetching image from {image_data_or_url}: {e}")
                    artwork_image = None
            elif isinstance(image_data_or_url, Image.Image):
                artwork_image = image_data_or_url
            else:
                artwork_image = None

            if artwork_image:
                try:
                    img_base64 = encode_image_to_base64(artwork_image)
                except Exception as e:
                    print(f"Error encoding image to base64: {e}")
                    img_base64 = None
            else:
                img_base64 = None

            # Fetch the timestamp of the like
            like_timestamp = user_likes[
                (user_likes['UserEmail'] == current_user_email) &
                (user_likes['ImageIndex'] == idx)
            ]['LikedAt'].iloc[0]
            like_timestamp_iso = like_timestamp.isoformat()

            liked_images.append({
                'index': int(idx),  # Convert to int
                'artist': curr_metadata['artist'],
                'style': curr_metadata['style'],
                'genre': curr_metadata['genre'],
                'description': f"{curr_metadata['genre']}, {curr_metadata['style']}",
                'image': img_base64,
                'timestamp': like_timestamp_iso  # ISO formatted timestamp
            })

        return jsonify({'liked_images': liked_images}), 200

    except Exception as e:
        return jsonify({'message': f'Error retrieving liked images: {str(e)}'}), 500

# --- Additional Initialization for image_likes.xlsx ---
# Initialize image_likes.xlsx entries for all images if not already present
try:
    if all_embeddings is not None and train_data is not None:
        image_likes = load_image_likes()
        existing_indices = set(image_likes['ImageIndex'].tolist())
        all_indices = set(range(all_embeddings.shape[0]))
        missing_indices = all_indices - existing_indices

        if missing_indices:
            new_entries = pd.DataFrame({
                'ImageIndex': list(missing_indices),
                'Users': [[] for _ in range(len(missing_indices))]  # Initialize with empty lists
            })
            image_likes = pd.concat([image_likes, new_entries], ignore_index=True)
            save_image_likes(image_likes)
            print(f"Initialized likes for {len(missing_indices)} images.")
        else:
            print("All images already have like entries.")
    else:
        print("Embeddings or training data not available. Skipping image likes initialization.")
except Exception as e:
    print(f"Error initializing image likes: {e}")

if __name__ == '__main__':
    app.run(debug=True)
