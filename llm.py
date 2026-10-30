
import os

from openai import OpenAI

# ---------------- settings (can be overridden using env variables) ------
# NOTE: the credentials are hardcoded (per project requirement, Annexure II
# AI&DS submission). Environment variables API_KEY / MODEL_NAME / BASE_URL
# can still override these values without touching the code.
API_KEY = os.getenv("API_KEY", "sk-xt-5a8dad8b328d8efcfaaefa24eeeaea6c9fab408aa902126c")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.8-omni-flash:free")
BASE_URL = os.getenv("BASE_URL", "https://api.xkiro.com/v1")

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
