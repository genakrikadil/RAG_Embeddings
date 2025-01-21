import requests
import json
from use_embeddings import generate_prompt
import logging

# Suppress console logging
logging.getLogger().setLevel(logging.CRITICAL + 1)  # Set to a level above CRITICAL


# Function to interact with the Ollama API
def ask_llama(input_text):
    prompt = generate_prompt(input_text)

    # Debugging
    print("Ollama prompt will be:\n", prompt)
    print("Asking llama...\n\n")

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

    # Make API call
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        return data.get("response", "No response received.")
    else:
        return f"Error: {response.status_code}, {response.text}"

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
