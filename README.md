# Project Overview

This project explores the implementation of Retrieval-Augmented Generation (RAG) using the Ollama server and its capabilities. The primary goal is to investigate how embeddings can improve interactions with the Ollama model.

The dataset used is Bible Trivia, consisting of 1,290 Q&A pairs in JSON format. The instruction fields contain questions, and the response fields provide concise answers.
Source: Bible Trivia Alpaca Dataset https://huggingface.co/datasets/liaaron1/bibile_trivia_alpaca

This idea was inspired by the article:
"When Llama Learns Bible: An End-to-End Walkthrough on Fine-Tuning the Llama 2 7B Model with Custom Data and Running It in Your Own Data Center"
Source: Medium Article by LichenLC "https://medium.com/@lichenlc/when-llama-learns-bible-ae550332fe8f"

Since model training is a resource-intensive process, I wondered if there might be another way. I explored the possibility of combining a JSON-based "training data" set with RAG to achieve similar results. Using a JSON database enables easy and rapid updates to the "Knowledge Database" of the expert system, allowing real-time updates without retraining and enabling agile modifications. The retrieved data can then be fed into a lightweight generative model to produce contextual responses.

The idea is to create a foundation for an "expert system" by providing Q&A pairs and leveraging Ollama as a language model to generate answers in a more humanized form.

It worked quite good in my opinion and can be used as "foundation" for another "AI Powered Expert System" project such as my "wendy-bot" project which you can find here: https://github.com/genakrikadil/wendy_addams_bot

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
    ├── use_FAISS.txt
    └── use_embeddings.py (this script reads "embeddings.db")


## How It Works

### Step 1: Prepare the Database
1. Run `import_jsonl.py`.  
   - This script reads `bible_trivia_alpaca.jsonl`.  
   - It generates embeddings and saves them into `embeddings.db`.

### Step 2: Process User Prompts
1. Use `use_embeddings.py`.  
   - This script takes a user prompt, retrieves the relevant embedding, and prepares a query for the Ollama server. It gives top N answers and its "Similarity Score"
   It downloads the whole SQL database into the memory and performs search line-by-line
2. Use `use_FAISS.py`.  
   - This script takes a user prompt, retrieves the relevant embedding, and prepares a query for the Ollama server. It gives top N answers and its "Similarity Score"
   it uses FAISS index to find "best records" in the Database
### Step 3: Generate Responses
1. Execute `ask_llama.py`.  
   - This script combines the user's question and the embeddings to generate a response from the Ollama server.
   You may select which way to use to process User's Input- Direct Search or FAISS

   Un-comment one of these in the ask_llama.py:

   #from use_embeddings import generate_prompt
   from use_FAISS import generate_prompt   
---

## Key Features

- Demonstrates a simple yet practical implementation of RAG.
- Highlights how embeddings can enhance real-life applications.
- Uses SQLite3 for lightweight, local storage of embeddings.
---

## Notes

- This project is a basic implementation designed for educational purposes.
- It provides a hands-on example of how embeddings integrate into an AI workflow.




