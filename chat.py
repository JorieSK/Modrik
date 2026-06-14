import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load .env so OPENAI_API_KEY is available in the environment
load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant. Keep answers concise."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


def chat(user_message, history):
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Replay prior turns so the model has context.
    for role, content in (history or []):
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return response.content