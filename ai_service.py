import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def stream_response(prompt: str, context: str = ""):
    client = OpenAI(api_key=os.getenv("API_KEY"))

    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            yield delta.content
