from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model =  ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature=0,
)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state : ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages' : [response]}


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

ck = InMemorySaver()
chatbot = graph.compile(checkpointer=ck)

