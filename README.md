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

# Why RAG? 

Leveraging Retrieval-Augmented Generation (RAG) offers several compelling benefits, including enhanced accuracy and relevance, customization, flexibility, and the ability to extend the model’s knowledge beyond its training data. Here’s a closer look at these advantages:

1. Improved Accuracy and Relevance
RAG significantly enhances the precision and pertinence of LLM-generated responses. By retrieving and integrating specific information from databases or datasets—often in real time—RAG ensures that outputs are grounded in both the model's pre-existing knowledge and the most up-to-date, relevant data you provide.

2. Customization
RAG empowers you to tailor the model’s responses to your specific domain or use case. By directing RAG to databases or datasets relevant to your application, you can fine-tune the model’s outputs to align closely with your needs, ensuring the information and tone are both targeted and useful.

3. Flexibility
RAG provides remarkable flexibility in accessing diverse data sources. Whether working with structured databases, unstructured web pages, or document repositories, RAG allows you to integrate various types of information. You can also update or replace data sources as needed, enabling the model to adapt seamlessly to dynamic information landscapes.

4. Expanding the Model’s Knowledge Beyond Training Data
LLMs are inherently limited by the scope of their training data. RAG overcomes this constraint by granting the model access to external information not included during its initial training. This expands the model’s knowledge base without requiring retraining, making it more versatile and responsive to new domains or rapidly changing topics.

5. Mitigating Hallucinations
A well-designed RAG system effectively minimizes the occurrence of hallucinations—instances where the model generates incorrect, fabricated, or nonsensical information. These errors can often appear convincingly phrased, making them difficult to detect. By relying on real-time retrieval of verified data, RAG can substantially reduce hallucinations, ensuring higher-quality and more reliable outputs.

These advantages highlight why RAG can be a transformative addition to your organization’s AI strategy. Next, let’s explore some of the challenges you may encounter when implementing RAG.


# This is an example how it works:

   **Enter your question (type 'exit' or 'quit' to stop): Why Moses crossed Red Sea?** 

   _# this text will be passed to Ollama to provide the answer_

   **Ollama prompt will be:**
   _Based on the user's input, please provide a precise answer by referencing the most similar records found in the database._

   **User Input: Why Moses crossed Red Sea? You must select the highest Similarity Score to response. Do not mention Response ID and Similarity Score in your response.**

   _Top 5 Most Similar Records:_
   1. Record ID: 4549 | Instruction: How did Moses command the Red Sea to divide so the Israelites could cross over? | Response: He lifted up his rod and stretched his hand over the sea (Exo 14:16,21) | Similarity Score: 0.8416
   2. Record ID: 3961 | Instruction: When the Egyptians tried to follow the Israelites through the Red Sea, what happened? | Response: The water crashed on them and killed them all. | Similarity Score: 0.6800
   3. Record ID: 4353 | Instruction: When Jesus walked on water, which sea was it? | Response: Sea of Galilee (John 6:1-19) | Similarity Score: 0.6646
   4. Record ID: 4223 | Instruction: Which sea did the Israelites cross through to escape the Egyptians? | Response: Red Sea (Exo 13:18) | Similarity Score: 0.6484
   5. Record ID: 4424 | Instruction: Who spotted Moses in the Nile placed in an ark of bulrushes? | Response: Pharaoh’s daughter (Exo 2:5) | Similarity Score: 0.6443

   _Please use the information above to generate a clear, accurate and direct response for the user based on the input provided. Do not repeat anything in this prompt._

 _# the end of text to Ollama_

   **Response from Llama:**

   **Moses crossed the Red Sea to allow the Israelites to escape from the pursuing Egyptians. The sea miraculously parted, allowing the Israelites to cross safely, while the pursuing forces of Egypt were destroyed by the returning waters.**

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
    ├── embeddings.db (DB file generated by import_jsonl.py)
    ├── faiss_index.bin (Index file created by import_jsonl.py)
    ├── import_jsonl.py (this script creates "embeddings.db" from JSONL)
    ├── README.md (this file)
    ├── requirements.txt (to use for .venv)
    ├── use_FAISS.py (generates ollama prompt based on data from .db)
    └── use_embeddings.py (generates ollama prompt based on data from .db)


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




