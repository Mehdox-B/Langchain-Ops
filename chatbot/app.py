# Define the required Packages and Modules
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Define the env variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# LangSmith Tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Define the Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your are a helpful assistant.Please give the user specific answer of his questions."),
        ("user", "Question:{question}")
    ]
)
# Streamlit Ui
st.title('Langchain-Ops Demo With OPENAI API')
input_text = st.text_input('Ask any Question about the topic you want ...')

# Invoke OpenAI_API
openai_llm = ChatOpenAI(model='gpt-3.5-turbo')
output_parser = StrOutputParser()
openai_chain = prompt | openai_llm | output_parser

if input_text:
    st.write(openai_chain.invoke({'question': input_text}))
