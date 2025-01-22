import requests
import json
import logging
# select if we want to use embeddings or FAISS
# FAISS is faster and uses indexing, "use_embeddings" 
# just loads database into the memory and searches for the closest embeddings
# using cosine similarity between the embeddings in Database and the input

#from use_embeddings import generate_prompt
from use_FAISS import generate_prompt


# Suppress console logging
logging.getLogger().setLevel(logging.CRITICAL + 1)  # Set to a level above CRITICAL


def ask_llama(input_text):
    prompt = generate_prompt(input_text)

    # Debugging
    print("Ollama prompt will be:\n", prompt)
    print("\n\n\n * * * Asking llama... * * * \n\n")

    # API request details
    url = "http://127.0.0.1:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }

    try:
        # Make API call
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # Check for a successful response
        if response.status_code == 200:
            response_text = response.text
            data = json.loads(response_text)
            return data.get("response", "No response received.")
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.ConnectionError:
        # Handle server not running or connection issues
        return "Error: Unable to connect to the Ollama server. Please ensure it is running."
    except Exception as e:
        # Catch other unexpected exceptions
        return f"An unexpected error occurred: {str(e)}"

# Main execution block##########################################
if __name__ == "__main__":
    while True:
        input_text = input("\n\nEnter your question (type 'exit' or 'quit' to stop): ")
        if input_text.lower() in {"exit", "quit"}:
            print("Exiting the program. Goodbye!")
            break
        
        # Get and print the response
        response = ask_llama(input_text)
        print("\nResponse from Llama:\n\n", response)
