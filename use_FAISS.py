import logging
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import faiss
import os
from sklearn.preprocessing import normalize

# Suppress INFO logs from sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# SQLite database file
db_file = 'embeddings.db'
faiss_index_file = 'faiss_index.bin'

# Initialize FAISS index
index = None
id_map = {}

# Create a SQLite database connection
def create_connection():
    conn = sqlite3.connect(db_file)
    return conn

# Build FAISS index and save to disk
def build_and_save_faiss_index():
    global index, id_map
    conn = create_connection()
    with conn:
        rows = conn.execute("SELECT id, embedding FROM resume_embeddings;").fetchall()

    # Prepare data for FAISS
    embeddings = []
    id_map = {}
    for idx, row in enumerate(rows):
        db_id, db_embedding = row
        db_embedding = np.frombuffer(db_embedding, dtype=np.float32)
        embeddings.append(db_embedding)
        id_map[idx] = db_id  # Map FAISS index to database ID

    # To ensure FAISS outputs cosine similarity scores, 
    # you need to normalize embeddings before adding them to the index:
    # Normalize embeddings
    
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = normalize(embeddings, axis=1, norm='l2')
    embedding_dim = embeddings.shape[1]  # Determine the dimension from data

    # Create FAISS index and add normalized embeddings
    index = faiss.IndexFlatIP(embedding_dim)  # Use Inner Product (dot product) for cosine similarity
    index.add(embeddings)


    # Save FAISS index to disk
    faiss.write_index(index, faiss_index_file)
    logger.info(f"FAISS index saved to {faiss_index_file}")

# Load FAISS index from disk
def load_faiss_index():
    global index, id_map
    if not os.path.exists(faiss_index_file):
        logger.error(f"FAISS index file {faiss_index_file} not found. Please build it first.")
        return False

    # Load the FAISS index
    index = faiss.read_index(faiss_index_file)
    logger.info(f"FAISS index loaded from {faiss_index_file}")

    # Reload id_map from database
    conn = create_connection()
    with conn:
        rows = conn.execute("SELECT id FROM resume_embeddings;").fetchall()
    id_map = {idx: row[0] for idx, row in enumerate(rows)}
    return True

# Query FAISS for nearest neighbors
def get_top_similar_faiss(input_text, top_n=5):
    global index, id_map
    if index is None:
        logger.error("FAISS index is not initialized. Run `load_faiss_index()` first.")
        return []

    input_embedding = model.encode([input_text])[0].astype(np.float32)
    distances, indices = index.search(np.array([input_embedding]), top_n)

    # Retrieve results
    results = []
    conn = create_connection()
    with conn:
        for i, idx in enumerate(indices[0]):
            db_id = id_map[idx]
            row = conn.execute("SELECT id, instruction, response FROM resume_embeddings WHERE id = ?", (db_id,)).fetchone()
            if row:
                db_id, db_instruction, db_response = row
                results.append((db_id, db_instruction, db_response, distances[0][i]))
    return results

# Generate a prompt for Ollama
def generate_prompt(input_text):
    # Check if FAISS index exists; if not, build it
    if not load_faiss_index():
        build_and_save_faiss_index()

    top_similarities = get_top_similar_faiss(input_text)
    prompt = (
        f"Based on the user's input, please provide a precise answer by referencing the most similar records found in the database.\n\n"
        f"User Input: {input_text} You must select the highest Similarity Score to response. Do not mention Response ID and Similarity Score in your response.\n\n"
        f"Top 5 Most Similar Records:\n"
    )
    for idx, record in enumerate(top_similarities, 1):
        db_id, instruction, response, distance = record
        prompt += f"{idx}. Record ID: {db_id} | Instruction: {instruction} | Response: {response} | Similarity Score: {distance:.4f}\n"
    prompt += (
        "\nPlease use the information above to generate a clear, accurate and direct response for the user based on the input provided. "
        "Do not repeat anything in this prompt."
    )
    return prompt

# Main logic for testing
if __name__ == "__main__":
 

    # User input for similarity search
    input_text = input("Enter text to find similar entries: ")
    prompt = generate_prompt(input_text)
    print("Ollama prompt will be:\n", prompt)
