import requests
from openai import OpenAI

# --- Ollama Local LLM Summarization ---
def summarize_with_ollama(transcript_text, model="llama3", url="http://localhost:11434/api/generate"):
    """
    Generates a summary using a local Ollama model.
    Assumes Ollama is running and the specified model is available.
    """
    prompt = f"Please provide a concise summary of the following meeting transcript:\n\n{transcript_text}"
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