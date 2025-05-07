# LangChain versions 0.0.x consisted of various conversational 
# memory types. Most of these are due for deprecation but still 
# hold value in understanding the different approaches that we can 
# take to building conversational memory.

# Throughout the notebook we will be referring to these older 
# memory types and then rewriting them using the recommended 
# RunnableWithMessageHistory class. We will learn about:

# ConversationBufferMemory: the simplest and most intuitive form of 
# conversational memory, keeping track of a conversation 
# without any additional bells and whistles.

# ConversationBufferWindowMemory: similar to ConversationBufferMemory, 
# but only keeps track of the last k messages.

# ConversationSummaryMemory: rather than keeping track of the entire 
# conversation, this memory type keeps track of a summary of the conversation.

# ConversationSummaryBufferMemory: merges the ConversationSummaryMemory and 
# ConversationTokenBufferMemory types.
# We'll work through each of these memory types in turn, and rewrite 
# each one using the RunnableWithMessageHistory class.



import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.messages import SystemMessage


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")



# 1. ConversationBufferMemory with RunnableWithMessageHistory

# ConversationBufferMemory is the simplest form of
# conversational memory, it is literally just a place
# that we store messages, and then use to feed messages into our LLM.

# Let's start with LangChain's original ConversationBufferMemory 
# object, we are setting return_messages=True to return the messages 
# as a list of ChatMessage objects — unless using a non-chat model 
# we would always set this to True as without it the messages are 
# passed as a direct string which can lead to unexpected behavior from chat LLMs.





# system_prompt = "You are a helpful assistant called Zeta."

# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(system_prompt),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("{query}"),
# ])

# pipeline = prompt_template | llm | {"res": lambda x: x.content}


# chat_map = {}
# def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = InMemoryChatMessageHistory()
#     return chat_map[session_id]




# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )

# res=pipeline_with_history.invoke(
#     {"query": "Hi, my name is James"},
#     config={"session_id": "id_123"}
# )

# print(res)

# res=pipeline_with_history.invoke(
#     {"query": "What is my name again?"},
#     config={"session_id": "id_123"}
# )

# print(res)


# 2. ConversationBufferWindowMemory with RunnableWithMessageHistory


# This all sync we can also make it async
# class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
#     messages: list[BaseMessage] = Field(default_factory=list)
#     k: int = Field(default_factory=int)

#     def __init__(self, k: int):
#         super().__init__(k=k)
#         print(f"Initializing BufferWindowMessageHistory with k={k}")

#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         """Add messages to the history, removing any messages beyond
#         the last `k` messages.
#         """
#         self.messages.extend(messages)
#         self.messages = self.messages[-self.k:]

#     def clear(self) -> None:
#         """Clear the history."""
#         self.messages = []

# chat_map = {}
# def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
#     print(f"get_chat_history called with session_id={session_id} and k={k}")
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = BufferWindowMessageHistory(k=k)
#     # remove anything beyond the last
#     return chat_map[session_id]



# system_prompt = "You are a helpful assistant called Zeta."

# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(system_prompt),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("{query}"),
# ])

# pipeline = prompt_template | llm | {"res": lambda x: x.content}

# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(
#             id="session_id",
#             annotation=str,
#             name="Session ID",
#             description="The session ID to use for the chat history",
#             default="id_default",
#         ),
#         ConfigurableFieldSpec(
#             id="k",
#             annotation=int,
#             name="k",
#             description="The number of messages to keep in the history",
#             default=4,
#         )
#     ]
# )

# # First, initialize the chat history by making an initial call
# res = pipeline_with_history.invoke(
#     {"query": "Hi, my name is James"},
#     config={"configurable": {"session_id": "id_k4", "k": 16}}
# )
# chat_map["id_k4"].clear()

# # Now we can add messages to the existing chat history
# chat_map["id_k4"].add_user_message("Hi, my name is James")
# chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
# chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
# chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
# chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
# chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
# chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
# chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
# chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
# chat_map["id_k4"].add_ai_message("Very cool!")

# # Query with k=4 (keeping last 4 messages)
# res = pipeline_with_history.invoke(
#     {"query": "what is my name again?"},
#     config={"configurable": {"session_id": "id_k4", "k": 4}}
# )
# print(res)


# 3. ConversationSummaryMemory with RunnableWithMessageHistory

# class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
#     messages: list[BaseMessage] = Field(default_factory=list)
#     llm: ChatOpenAI = Field(default_factory=ChatOpenAI)

#     def __init__(self, llm: ChatOpenAI):
#         super().__init__(llm=llm)

#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         """Add messages to the history and create a summary."""
#         # Get existing summary if any
#         existing_summary = ""
#         if self.messages:
#             existing_summary = self.messages[0].content

#         # Format new messages
#         new_messages_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

#         # Create summary prompt
#         summary_prompt = ChatPromptTemplate.from_messages([
#             SystemMessagePromptTemplate.from_template(
#                 "Given the existing conversation summary and the new messages, "
#                 "generate a new summary of the conversation. Ensuring to maintain "
#                 "as much relevant information as possible."
#             ),
#             HumanMessagePromptTemplate.from_template(
#                 "Existing conversation summary:\n{existing_summary}\n\n"
#                 "New messages:\n{messages}"
#             )
#         ])

#         # Generate new summary
#         new_summary = self.llm.invoke(
#             summary_prompt.format_messages(
#                 existing_summary=existing_summary,
#                 messages=new_messages_text
#             )
#         )

#         # Replace the existing history with a single system summary message
#         self.messages = [SystemMessage(content=new_summary.content)]

#     def clear(self) -> None:
#         """Clear the history."""
#         self.messages = []


# chat_map = {}
# def get_chat_history(session_id: str, llm: ChatOpenAI) -> ConversationSummaryMessageHistory:
#     if session_id not in chat_map:
#         # if session ID doesn't exist, create a new chat history
#         chat_map[session_id] = ConversationSummaryMessageHistory(llm=llm)
#     # return the chat history
#     return chat_map[session_id]

# system_prompt = "You are a helpful assistant called Zeta."

# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(system_prompt),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("{query}"),
# ])

# pipeline = prompt_template | llm | {"res": lambda x: x.content}

# pipeline_with_history = RunnableWithMessageHistory(
#     pipeline,
#     get_session_history=get_chat_history,
#     input_messages_key="query",
#     history_messages_key="history",
#     history_factory_config=[
#         ConfigurableFieldSpec(
#             id="session_id",
#             annotation=str,
#             name="Session ID",
#             description="The session ID to use for the chat history",
#             default="id_default",
#         ),
#         ConfigurableFieldSpec(
#             id="llm",
#             annotation=ChatOpenAI,
#             name="LLM",
#             description="The LLM to use for the conversation summary",
#             default=llm,
#         )
#     ]
# )


# res=pipeline_with_history.invoke(
#     {"query": "Hi, my name is James and my friend is Aditya"},
#     config={"session_id": "id_123", "llm": llm}
# )
# res=pipeline_with_history.invoke(
#     {"query": "I'm researching the different types of conversational memory."},
#     config={"session_id": "id_123", "llm": llm}
# )
# res=pipeline_with_history.invoke(
#     {"query": "What is the capital of France?"},
#     config={"session_id": "id_123", "llm": llm}
# )
# res=pipeline_with_history.invoke(
#     {"query": "Tell me about best food in delhi"},
#     config={"session_id": "id_123", "llm": llm}
# )
# res=pipeline_with_history.invoke(
#     {"query": "Tell me about python"},
#     config={"session_id": "id_123", "llm": llm}
# )
# res=pipeline_with_history.invoke(
#     {"query": "Tell my friend's name"},
#     config={"session_id": "id_123", "llm": llm}
# )
# print(res)

# 4. ConversationSummaryBufferMemory with RunnableWithMessageHistory

class ConversationSummaryBufferMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI = Field(default_factory=ChatOpenAI)
    k: int = Field(default_factory=int)

    def __init__(self, llm: ChatOpenAI, k: int):
        super().__init__(llm=llm, k=k)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages and summarizing the messages that we
        drop.
        """
        existing_summary: SystemMessage | None = None
        old_messages: list[BaseMessage] | None = None
        # see if we already have a summary message
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            print(">> Found existing summary")
            existing_summary = self.messages.pop(0)
        # add the new messages to the history
        self.messages.extend(messages)
        # check if we have too many messages
        if len(self.messages) > self.k:
            print(
                f">> Found {len(self.messages)} messages, dropping "
                f"oldest {len(self.messages) - self.k} messages.")
            # pull out the oldest messages...
            old_messages = self.messages[:self.k]
            # ...and keep only the most recent messages
            self.messages = self.messages[-self.k:]
        if old_messages is None:
            print(">> No old messages to update summary with")
            # if we have no old_messages, we have nothing to update in summary
            return
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible. However we want to keep the conversation short and concise."
                "the limmit is single short paragraph"
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])
        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                old_messages=old_messages
            )
        )
        print(f">> New summary: {new_summary.content}")
        # prepend the new summary to the history
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

chat_map = {}
def get_chat_history(session_id: str, llm: ChatOpenAI, k: int) -> ConversationSummaryBufferMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryBufferMessageHistory(llm=llm, k=k)
    # return the chat history
    return chat_map[session_id]

system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

pipeline = prompt_template | llm | {"res": lambda x: x.content}

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="llm",
            annotation=ChatOpenAI,
            name="LLM",
            description="The LLM to use for the conversation summary",
            default=llm,
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

pipeline_with_history.invoke(
    {"query": "Hi, my name is James"},
    config={"session_id": "id_123", "llm": llm, "k": 4}
)
chat_map["id_123"].messages


for i, msg in enumerate([
    "I'm researching the different types of conversational memory.",
    "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory.",
    "Buffer memory just stores the entire conversation",
    "Buffer window memory stores the last k messages, dropping the rest."
]):
    print(f"---\nMessage {i+1}\n---\n")
    pipeline_with_history.invoke(
        {"query": msg},
        config={"session_id": "id_123", "llm": llm, "k": 4}
    )