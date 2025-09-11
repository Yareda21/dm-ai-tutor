from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=os.getenv("SECOND_OPEN_AI_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=4096,
)

result = llm.invoke("hi who are you")
print(result)