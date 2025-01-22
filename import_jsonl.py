# read from jsonl file "bible_trivia_alpaca.jsonl" and create embeddings
# file embeddings.db
# these embeddings will be used later on to create a chatbot
#

import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import json


# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')



# Initialize SQLite database
def initialize_database(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Create the table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS resume_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instruction TEXT,
            response TEXT,
            embedding BLOB
        );
        """)
        conn.commit()
        print("Database initialized successfully.")
        cur.close()
        conn.close()
    except Exception as e:
        print("Error initializing database:", e)

# Insert embeddings into SQLite
def insert_embeddings(db_path, instructions, contexts, responses, embeddings):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Insert vector data
        for instruction, context, response, embedding in zip(instructions, contexts, responses, embeddings):
            cur.execute(
                "INSERT INTO resume_embeddings (instruction, context, response, embedding) VALUES (?, ?, ?, ?)",
                (instruction, context, response, embedding.tobytes())
            )
        conn.commit()
        print("Data has been vectorized and successfully inserted into the database.")
        print(f"Number of entries: {len(instructions)}")
        print(f"Number of generated vectors: {len(embeddings)}")

        cur.close()
        conn.close()
    except Exception as e:
        print("Error inserting embeddings:", e)

# Query embeddings from SQLite (optional)
def fetch_embeddings(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT instruction, context, response, embedding FROM resume_embeddings;")
        rows = cur.fetchall()

        data = []
        for row in rows:
            instruction = row[0]
            context = row[1]
            response = row[2]
            embedding = np.frombuffer(row[3], dtype=np.float32)
            data.append((instruction, context, response, embedding))

        print("Data fetched successfully.")
        cur.close()
        conn.close()
        return data
    except Exception as e:
        print("Error fetching embeddings:", e)
        return []

# Main logic
if __name__ == "__main__":
    # Database file path
    db_path = "embeddings.db"
    json_file_path = 'bible_trivia_alpaca.jsonl'  # Path to your JSON file

    # Step 1: Initialize SQLite database
    initialize_database(db_path)

    # Step 2: Load and process JSON data
    with open(json_file_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    instructions = [item['instruction'] for item in data]
    contexts = [item['context'] for item in data]
    responses = [item['response'] for item in data]

    print(f"Loaded {len(instructions)} entries from the JSON file.")

    # Step 3: Convert to vectors (only use instruction and response for embeddings)
    texts_to_embed = instructions + responses  # Combining instruction and response
    embeddings = model.encode(texts_to_embed)

    # Step 4: Insert data into the database
    insert_embeddings(db_path, instructions, contexts, responses, embeddings)

    # Optional: Fetch and display embeddings
    fetched_data = fetch_embeddings(db_path)
    for instruction, context, response, embedding in fetched_data[:5]:  # Show only the first 5 for brevity
        print(f"Instruction: {instruction}\nContext: {context}\nResponse: {response}\nEmbedding: {embedding[:10]}... (truncated)\n")
