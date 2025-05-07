from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")




@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


# For getting the information
print(add.args_schema.model_json_schema())





prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

memory = ConversationBufferMemory(
    memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
    return_messages=True  # to return Message objects
)



tools = [add, subtract, multiply, exponentiate]

agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
res=agent_executor.invoke({
    "input": "what is 10.7 multiplied by 7.68?",
    "chat_history": memory.chat_memory.messages,
})
print(res)

res=agent_executor.invoke({
    "input": "My name is James",
    "chat_history": memory
})

print(res)

res=agent_executor.invoke({
    "input": "What is nine plus 10, minus 4 * 2, to the power of 3",
    "chat_history": memory
})

print(res)

res=agent_executor.invoke({
    "input": "What is my name",
    "chat_history": memory
})

print(res)