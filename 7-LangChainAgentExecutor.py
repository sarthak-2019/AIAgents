from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages import ToolMessage
import json
import os
load_dotenv()
# Set API keys
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

# Define the multiply tool
@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

# Define the exponentiate tool
@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

@tool
def final_answer(answer: str, tools_used: list[str]) -> str:
    """Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    return {"answer": answer, "tools_used": tools_used}



# Creating an Agent

# We will use LangChain Epression Language (LCEL)
# to construct the agent. We will cover LCEL more 
# in the next chapter, but for now - all we need to 
# know is that our agent will be constructed using syntax 
# and components like so:

# agent = (
#     <input parameters, including chat history and user query>
#     | <prompt>
#     | <LLM with tools>
# )





prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided in the "
        "'scratchpad' below. If you have an answer in the "
        "scratchpad you should not use any more tools and "
        "instead answer directly to the user."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# To add tools to our LLM, we will use the 
# bind_tools method within the LCEL constructor, 
# which will take our tools and add them to the LLM. 
# We'll also include the tool_choice="any" argument 
# to bind_tools, which tells the LLM that it MUST use 
# a tool, ie it cannot provide a final answer directly
# (in therefore not using a tool):



tools = [final_answer,add, subtract, multiply, exponentiate]

# define the agent runnable
agent: RunnableSerializable = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

tool_call = agent.invoke({"input": "What is 10 + 10", "chat_history": []})


# From here, we have the tool name that our LLM wants to 
# use and the args that it wants to pass to that tool. 
# We can see that the tool add is being used with the arguments 
# x=10 and y=10. The agent.invoke method has not executed the 
# tool function; we need to write that part of the agent code ourselves.

# Executing the tool code requires two steps:

# Map the tool name to the tool function.

# Execute the tool function with the generated args.


# create tool name to function mapping
name2tool = {tool.name: tool.func for tool in tools}

tool_exec_content = name2tool[tool_call.tool_calls[0]["name"]](
    **tool_call.tool_calls[0]["args"]
)
print(tool_exec_content)

tool_exec = ToolMessage(
    content=f"The {tool_call.tool_calls[0]['name']} tool returned {tool_exec_content}",
    tool_call_id=tool_call.tool_calls[0]["id"]
)

out = agent.invoke({
    "input": "What is 10 + 10",
    "chat_history": [],
    "agent_scratchpad": [tool_call, tool_exec]
})
print(out.tool_calls[0]["args"])

# We can set tool_choice="any" to tell the LLM that it MUST use a tool.
# We can also set tool_choice="required" to tell the LLM that it MUST use a tool,
# and if it cannot find a tool to use, it should return an error.

# If we set it to "auto" (the default), the LLM will use a tool if it can,
# but if it cannot find a tool to use, it will return an error.



# Building a Custom Agent Execution Loop


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class CustomAgentExecutor:
    chat_history: list[BaseMessage]

    def __init__(self, max_iterations: int = 3):
        self.chat_history = []
        self.max_iterations = max_iterations
        self.agent: RunnableSerializable = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")  # we're forcing tool use again
        )

    def invoke(self, input: str) -> dict:
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        agent_scratchpad = []
        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_call = self.agent.invoke({
                "input": input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            # add initial tool call to scratchpad
            agent_scratchpad.append(tool_call)
            # otherwise we execute the tool and add it's output to the agent scratchpad
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_call_id = tool_call.tool_calls[0]["id"]
            tool_out = name2tool[tool_name](**tool_args)
            # add the tool output to the agent scratchpad
            tool_exec = ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            agent_scratchpad.append(tool_exec)
            # add a print so we can see intermediate steps
            print(f"{count}: {tool_name}({tool_args})")
            count += 1
            # if the tool call is the final answer tool, we stop
            if tool_name == "final_answer":
                break
        # add the final output to the chat history
        final_answer = tool_out["answer"]
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer)
        ])
        # return the final answer in dict form
        return json.dumps(tool_out)

agent_executor = CustomAgentExecutor()
agent_executor.invoke(input="What is 10 + 10")
