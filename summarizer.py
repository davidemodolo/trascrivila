import requests
from openai import OpenAI

# --- Ollama Local LLM Summarization ---
def summarize_with_ollama(transcript_text, model="gemma3:12b", url="http://localhost:11434/api/generate"):
    """
    Generates a summary using a local Ollama model.
    Assumes Ollama is running and the specified model is available.
    """
    prompt = f"Please provide a concise summary of the following meeting transcript and end the summary with a bulleted list on what we have to do:\n\n{transcript_text}"
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120 # Set a timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Assuming the response is a JSON object with a 'response' key
        summary = response.json().get("response", "No summary content returned.")
        return summary, None
    except requests.exceptions.ConnectionError:
        return None, "Ollama connection failed. Is Ollama running at http://localhost:11434?"
    except requests.exceptions.RequestException as e:
        return None, f"An error occurred with Ollama request: {e}"

# --- OpenAI GPT Summarization ---
def summarize_with_openai(transcript_text, api_key):
    """
    Generates a summary using the OpenAI API (GPT model).
    """
    if not api_key:
        return None, "OpenAI API key is missing."
        
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
                {"role": "user", "content": f"Please provide a concise summary of the following meeting transcript:\n\n{transcript_text}"}
            ]
        )
        summary = response.choices[0].message.content
        return summary, None
    except Exception as e:
        return None, f"An error occurred with OpenAI API: {e}"
    
if __name__ == "__main__":
    # Example usage
    transcript = "This is a sample meeting transcript that needs to be summarized."
    
    # Summarize using Ollama
    ollama_summary, ollama_error = summarize_with_ollama(transcript)
    if ollama_error:
        print(f"Ollama Error: {ollama_error}")
    else:
        print(f"Ollama Summary: {ollama_summary}")
    
    # Summarize using OpenAI
    openai_api_key = "your_openai_api_key_here"  # Replace with your actual OpenAI API key
    openai_summary, openai_error = summarize_with_openai(transcript, openai_api_key)
    if openai_error:
        print(f"OpenAI Error: {openai_error}")
    else:
        print(f"OpenAI Summary: {openai_summary}")