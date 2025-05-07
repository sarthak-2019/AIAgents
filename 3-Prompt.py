
# We'll start by looking at the various parts of our prompt. For RAG use-cases we'll 
# typically have three core components however this is very use-cases dependant 
# and can vary significantly. Nonetheless, for RAG we will typically see:

# Rules for our LLM: this part of the prompt sets up the behavior of our LLM,
#  how it should approach responding to user queries, and simply providing as much
# information as possible about what we're wanting to do as possible. 
# We typically place this within the system prompt of an chat LLM.

# Context: this part is RAG-specific. The context refers to some external information 
# that we may have retrieved from a web search, database query, or often a vector database. 
# This external information is the Retrieval Augmentation part of RAG. For chat LLMs we'll 
# typically place this inside the chat messages between the assistant and user.

# Question: this is the input from our user. In the vast majority of cases 
# the question/query/user input will always be provided to the LLM 
# (and typically through a user message). However, the format and 
# location of this being provided often changes.

# Answer: this is the answer from our assistant, again this 
# is very typical and we'd expect this with every use-case.

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

prompt = """
Answer the user's query based on the context below.
If you cannot answer the question using the
provided information answer with "I don't know".

Context: {context}
"""


# passing the template to the LangChain model
prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("user", "{query}"),
])


prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt),
    HumanMessagePromptTemplate.from_template("{query}"),
])


# To check all the variables
print(prompt_template.input_variables)

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
# Correct model name
openai_model = "gpt-4o-mini"

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model=openai_model)

# pipeline = (
#     {
#         "query": lambda x: x["query"],
#         "context": lambda x: x["context"]
#     }
#     | prompt_template
#     | llm
#     | {"response": lambda x: x.content}
# )

context = """Aurelio AI is an AI company developing tooling for AI
engineers. Their focus is on language AI with the team having strong
expertise in building AI agents and a strong background in
information retrieval.

The company is behind several open source frameworks, most notably
Semantic Router and Semantic Chunkers. They also have an AI
Platform providing engineers with tooling to help them build with
AI. Finally, the team also provides development services to other
organizations to help them bring their AI tech to market.

Aurelio AI became LangChain Experts in September 2024 after a long
track record of delivering AI solutions built with the LangChain
ecosystem."""

query = "what does Aurelio AI do?"

# res=pipeline.invoke({"query": query, "context": context})

# print(res)



# Few Shot Prompting


# examples = [
#     {
#         "input": "Can you explain gravity?",
#         "output": (
#             "## Gravity\n\n"
#             "Gravity is one of the fundamental forces in the universe.\n\n"
#             "### Discovery\n\n"
#             "* Gravity was first discovered by Sir Isaac Newton in the late 17th century.\n"
#             "* It was said that Newton theorized about gravity after seeing an apple fall from a tree.\n\n"
#             "### In General Relativity\n\n"
#             "* Gravity is described as the curvature of spacetime.\n"
#             "* The more massive an object is, the more it curves spacetime.\n"
#             "* This curvature is what causes objects to fall towards each other.\n\n"
#             "### Gravitons\n\n"
#             "* Gravitons are hypothetical particles that mediate the force of gravity.\n"
#             "* They have not yet been detected.\n\n"
#             "**To conclude**, Gravity is a fascinating topic and has been studied extensively since the time of Newton.\n\n"
#         )
#     },
#     {
#         "input": "What is the capital of France?",
#         "output": (
#             "## France\n\n"
#             "The capital of France is Paris.\n\n"
#             "### Origins\n\n"
#             "* The name Paris comes from the Latin word \"Parisini\" which referred to a Celtic people living in the area.\n"
#             "* The Romans named the city Lutetia, which means \"the place where the river turns\".\n"
#             "* The city was renamed Paris in the 3rd century BC by the Celtic-speaking Parisii tribe.\n\n"
#             "**To conclude**, Paris is highly regarded as one of the most beautiful cities in the world and is one of the world's greatest cultural and economic centres.\n\n"
#         )
#     }
# ]
# new_system_prompt = """
# Answer the user's query based on the context below.
# If you cannot answer the question using the
# provided information answer with "I don't know".

# Always answer in markdown format. When doing so please
# provide headers, short summaries, follow with bullet
# points, then conclude.

# Context: {context}
# """

# example_prompt = ChatPromptTemplate.from_messages([
#     ("human", "{input}"),
#     ("ai", "{output}"),
# ])

# print(example_prompt)
# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", new_system_prompt),
#     few_shot_prompt,
#     ("user", "{query}"),
# ])

# pipeline = prompt_template | llm
# out = pipeline.invoke({"query": query, "context": context}).content
# print(out)




# Chain of Thought Prompting

# We'll take a look at one more commonly used prompting technique called chain 
# of thought (CoT). CoT is a technique that encourages the LLM to think through 
# the problem step by step before providing an answer. The idea being that by 
# breaking down the problem into smaller steps, the LLM is more likely to 
# arrive at the correct answer and we are less likely to see hallucinations.

# To implement CoT we don't need any specific LangChain objects, 
# instead we are simply modifying how we instruct our LLM within the system
# prompt. We will ask the LLM to list the problems that need to be solved,
# to solve each problem individually, and then to arrive at the final answer.


# Define the chain-of-thought prompt template
cot_system_prompt = """
Be a helpful assistant and answer the user's question.

To answer the question, you must:

- List systematically and in precise detail all
  subproblems that need to be solved to answer the
  question.
- Solve each sub problem INDIVIDUALLY and in sequence.
- Finally, use everything you have worked through to
  provide the final answer.
"""
query = (
    "How many keystrokes are needed to type the numbers from 1 to 500?"
)


cot_prompt_template = ChatPromptTemplate.from_messages([
    ("system", cot_system_prompt),
    ("user", "{query}"),
])

cot_pipeline = cot_prompt_template | llm


cot_result = cot_pipeline.invoke({"query": query}).content
print(cot_result)