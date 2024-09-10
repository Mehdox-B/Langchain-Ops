import requests
import json
import streamlit as st
from typing import Union
from langchain_core.messages import HumanMessage,AIMessage
from fastapi import FastAPI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import uvicorn
from langchain_community.llms import openai
from app import get_response_from_chain

##-----------Building The Client UI-------------##
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[AIMessage(content="Hello Highness Employer , Ask me Anything About Marketing,Seo, ...")]

##-------Calling the api endpoint-------------##
async def get_openai_response(query:str):
    response= await requests.post("http://localhost:9090/openai/stream",
    json={'input':{'domain':query}})
    return response['data'][0]['content']

st.set_page_config(page_title="Q&A Chatbot",page_icon="💬")
st.title("Marketing Expert Q&A ChatBot 🎯")

##---------------conversation-------------##
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

##-------------user question/query-------------##
user_question = st.chat_input(placeholder="Your message/question ...")
if user_question is not None and user_question !="":
    st.session_state.chat_history.append(HumanMessage(content=user_question))

    with st.chat_message("human"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        #assistant_response = openai_model.stream(user_question)
        #jsondata = json.dump(fp=conversation.stream(user_question))
        assistant_response =  get_response_from_chain(user_question,st.session_state.chat_history)
        ai_message = st.write_stream(assistant_response)
        
    
    st.session_state.chat_history.append(AIMessage(content=ai_message))

##-----------Notes About This ChatBot------------------##

#It Will Stream the responses from the LLM as it is being generated.
#It will use Langchain Wrapper to interact with LLM Models.
#It Will use streamlit to create the UI-System.
#It will use memory mechanism to remember the chat history and show it to the user.
