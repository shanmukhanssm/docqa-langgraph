
import os

from openai import OpenAI

# ---------------- settings (change using env variables) ----------------
API_KEY = os.getenv("API_KEY", "not_set_yet")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")

# one global client, we reuse the same client everywhere in the project.
# max_retries makes ask_llm survive transient errors like HTTP 429
# (rate limit) or 5xx - the SDK retries them with exponential backoff,
# which matters because summarizing a big document fires many calls.
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=4)


def ask_llm(prompt: str) -> str:
    """
    Sends one prompt to the LLM and returns the reply text.
    Kept super simple on purpose - the whole project uses only this
    one function to talk to the LLM.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful study assistant. Be short and clear."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # low temperature so answers stay focused on the document
    )
    return response.choices[0].message.content
