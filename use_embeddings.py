# Description: This script is used to find similar entries in the database based on user input.
# It uses the SentenceTransformer model to encode the user input and the stored embeddings, and then computes the cosine similarity between them.
# The top similar entries are then returned to the user.
# The script also generates a prompt for Ollama based on the top similar entries found.
#
# pip install "urllib3<2"
#

import logging
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Suppress INFO logs from sentence_transformers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize SQLite database
db_file = 'embeddings.db'

# Create a SQLite database connection
def create_connection():
    conn = sqlite3.connect(db_file)
    return conn

# Retrieve all embeddings from the database
def get_all_embeddings():
    conn = create_connection()
    with conn:
        rows = conn.execute("SELECT id, instruction, response, embedding FROM resume_embeddings;").fetchall()
    conn.close()
    return rows

# Function to compute similarity between input and stored embeddings
def get_top_similar(input_text, top_n=5):
    #logger.info(f"USER INPUT: {input_text}")
    input_embedding = model.encode([input_text])[0]
    all_similarities = []

    try:
        rows = get_all_embeddings()

        for row in rows:
            db_id, db_instruction, db_response, db_embedding = row
            db_embedding = np.frombuffer(db_embedding, dtype=np.float32)

            similarity = cosine_similarity([input_embedding], [db_embedding])[0][0]
            all_similarities.append((db_id, db_instruction, db_response, similarity))

        all_similarities.sort(key=lambda x: x[3], reverse=True)
        return all_similarities[:top_n]
    
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []
def generate_prompt(input_text):
    top_similarities = get_top_similar(input_text)
    prompt = (
        f"Based on the user's input, please provide a precise answer by referencing the most similar records found in the database.\n\n"
        f"User Input: {input_text} You must select the highest Similarity Score to response. Do not mention Response ID and Similarity Score in your response.\n\n"
        f"Top 5 Most Similar Records:\n"
    )
    for idx, record in enumerate(top_similarities, 1):
        db_id, instruction, response, sim = record
        prompt += f"{idx}. Record ID: {db_id} | Instruction: {instruction} | Response: {response} | Similarity Score: {sim:.4f}\n"
    prompt += (
        "\nPlease use the information above to generate a clear, accurate and direct response for the user based on the input provided. "
        "Do not repeat anything in this prompt. Use the highest Similarity Score to guide your response."
    )
    return prompt

# Main logic for testing #######################
if __name__ == "__main__":
    # User input for similarity search
    input_text = input("Enter text to find similar entries: ")
    prompt=generate_prompt(input_text)
    print("Ollama prompt will be:\n", prompt)
