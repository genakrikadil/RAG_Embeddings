# Project Overview

This project demonstrates the implementation of Retrieval-Augmented Generation (RAG) using the Ollama server and its capabilities. The primary focus is to explore how embeddings can enhance interactions with the Ollama model.

---

## Requirements

To run this project, you will need the following:

- **Python 3.9.21**  
  *(Other versions have not been tested.)*

- **Docker** or **Python Virtual Environment (venv)**

- **Ollama Server**  
  *(Configured with `llama3.2` image or any other image of your choice, adjust ask_llama.py for your model)*

---

## Project Structure

Below is an overview of the project directory:

    .
    ├── ask_llama.py (gets embeddings and send them to ollama)
    ├── bible_trivia_alpaca.jsonl (source file, exported into db)
    ├── embeddings.db
    ├── import_jsonl.py (this script creates "embeddings.db" from JSONL)
    ├── README.md
    ├── requirements.txt
    └── use_embeddings.py (this script reads "embeddings.db")


## How It Works

### Step 1: Prepare the Database
1. Run `import_jsonl.py`.  
   - This script reads `bible_trivia_alpaca.jsonl`.  
   - It generates embeddings and saves them into `embeddings.db`.

### Step 2: Process User Prompts
1. Use `use_embeddings.py`.  
   - This script takes a user prompt, retrieves the relevant embedding, and prepares a query for the Ollama server. It gives top N answers and its "Similarity Score"

### Step 3: Generate Responses
1. Execute `ask_llama.py`.  
   - This script combines the user's question and the embeddings to generate a response from the Ollama server.

---

## Key Features

- Demonstrates a simple yet practical implementation of RAG.
- Highlights how embeddings can enhance real-life applications.
- Uses SQLite3 for lightweight, local storage of embeddings.
---

## Notes

- This project is a basic implementation designed for educational purposes.
- It provides a hands-on example of how embeddings integrate into an AI workflow.




