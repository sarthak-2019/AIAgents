import os
from getpass import getpass
from dotenv import load_dotenv
from langsmith import traceable
import random
import time
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI

load_dotenv()


# must enter API key
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGSMITH_API_KEY')
# below should not be changed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# you can change this as preferred
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGSMITH_PROJECT") 

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# @traceable
# def generate_random_number():
#     return random.randint(0, 100)

# @traceable
# def generate_string_delay(input_str: str):
#     number = random.randint(1, 5)
#     time.sleep(number)
#     return f"{input_str} ({number})"

# @traceable
# def random_error():
#     number = random.randint(0, 1)
#     if number == 0:
#         raise ValueError("Random error")
#     else:
#         return "No error"



# for _ in tqdm(range(10)):
#     generate_random_number()
#     generate_string_delay("Hello")
#     try:
#         random_error()
#     except ValueError:
#         pass


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
    "Enter OpenAI API Key: "
)

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

llm.invoke("What is the capital of France?")