# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# artist_encoder = LabelEncoder()
# style_encoder = LabelEncoder()
# genre_encoder = LabelEncoder()

# artist_encoder.fit(train_data['artist'])
# style_encoder.fit(train_data['style'])
# genre_encoder.fit(train_data['genre'])

# def encode_metadata(example):
#     example['artist_encoded'] = artist_encoder.transform([example['artist']])[0]
#     example['style_encoded'] = style_encoder.transform([example['style']])[0]
#     example['genre_encoded'] = genre_encoder.transform([example['genre']])[0]
#     return example

# dataset_encoded = train_data.map(encode_metadata)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def generate_clip_embeddings(example):
#     if isinstance(example['image'], str):  # It's a file path
#         image = Image.open(example['image']).convert("RGB")
#     elif isinstance(example['image'], Image.Image):  # It's a PIL image
#         image = example['image'].convert("RGB")
#     else:
#         raise ValueError(f"Unexpected type for 'image': {type(example['image'])}")

#     inputs = clip_processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**inputs)

#     example['image_embeddings'] = image_features.cpu().numpy()
#     return example

# dataset_encoded = dataset_encoded.map(generate_clip_embeddings)

# def combine_embeddings(example):
#     # Ensure metadata is properly processed
#     metadata_vector = np.array([
#         example['artist_encoded'],
#         example['style_encoded'],
#         example['genre_encoded']
#     ], dtype=np.float32)

#     # Normalize metadata vector
#     metadata_vector = torch.nn.functional.normalize(
#         torch.tensor(metadata_vector), dim=0
#     ).numpy()

#     # Handle the case where image embeddings are stored as a list
#     image_embeddings = np.array(example['image_embeddings'])  # Convert to NumPy array if needed

#     # Combine embeddings
#     combined_embedding = np.concatenate([image_embeddings.flatten(), metadata_vector])
#     example['combined_embeddings'] = combined_embedding
#     return example

# dataset_encoded = dataset_encoded.map(combine_embeddings)

# np.save("combined_embeddings.npy", np.vstack(dataset_encoded['combined_embeddings']))

# # print("Embeddings prepared and saved successfully!")

# all_embeddings = np.array([example['combined_embeddings'] for example in dataset_encoded])