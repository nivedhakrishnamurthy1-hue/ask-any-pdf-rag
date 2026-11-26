"""Thin wrapper around the Groq chat API with JSON mode and retries."""
import os
import time


def get_client(api_key=None):
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    from groq import Groq  # imported lazily so tests don't need the package

    return Groq(api_key=key)


def complete_json(system, user, model, temperature=0.0, retries=2, api_key=None):
    client = get_client(api_key)
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        except Exception as error:  # rate limits, network blips
            last_error = error
            if attempt < retries:
                time.sleep(2 * (2 ** attempt))
    raise last_error
