import numpy as np
from image_engine import ImageEngine
from database import DatabaseManager

engine = ImageEngine()
db = DatabaseManager()

# Load all indexed images
image_data = db.fetch_all_images()
image_names = []
image_embeddings = []

for file_path, file_name, file_hash, last_modified, embedding in image_data:
    image_names.append(file_name)
    image_embeddings.append(embedding)

image_embeddings = np.array(image_embeddings)

# Test queries
queries = ["building", "blue shirt", "food", "person", "map", "computer"]

for query in queries:
    print(f"\n--- Query: '{query}' ---")
    query_vector = engine.text_to_vector(query)
    similarities = np.dot(image_embeddings, query_vector)

    scored = sorted(
        zip(image_names, similarities),
        key=lambda x: x[1],
        reverse=True
    )
    for name, score in scored[:5]:
        print(f"  {score:.4f}  {name}")